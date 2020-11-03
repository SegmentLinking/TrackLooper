#include "StudySDLTrackletDebugNtupleWriter.h"

StudySDLTrackletDebugNtupleWriter::StudySDLTrackletDebugNtupleWriter()
{
}

void StudySDLTrackletDebugNtupleWriter::bookStudy()
{
    ana.tx->createBranch<float>("pt");
    ana.tx->createBranch<float>("eta");
    ana.tx->createBranch<float>("dxy");
    ana.tx->createBranch<float>("dz");
    ana.tx->createBranch<float>("pdgid");
    ana.tx->createBranch<float>("itrk");
    ana.tx->createBranch<int  >("is_trk_bbbbbb");
    ana.tx->createBranch<int  >("is_trk_bbbbbe");
    ana.tx->createBranch<int  >("is_trk_bbbbee");
    ana.tx->createBranch<int  >("is_trk_bbbeee");
    ana.tx->createBranch<int  >("is_trk_bbeeee");
    ana.tx->createBranch<int  >("is_trk_beeeee");
    ana.tx->createBranch<vector<float>>("simhit_x");
    ana.tx->createBranch<vector<float>>("simhit_y");
    ana.tx->createBranch<vector<float>>("simhit_z");
    ana.tx->createBranch<vector<float>>("simhit_px");
    ana.tx->createBranch<vector<float>>("simhit_py");
    ana.tx->createBranch<vector<float>>("simhit_pz");
    ana.tx->createBranch<vector<float>>("ph2_x");
    ana.tx->createBranch<vector<float>>("ph2_y");
    ana.tx->createBranch<vector<float>>("ph2_z");

    ana.tx->createBranch<vector<float>>("simhit_x_pdgid_matched");
    ana.tx->createBranch<vector<float>>("simhit_y_pdgid_matched");
    ana.tx->createBranch<vector<float>>("simhit_z_pdgid_matched");
    ana.tx->createBranch<vector<float>>("simhit_px_pdgid_matched");
    ana.tx->createBranch<vector<float>>("simhit_py_pdgid_matched");
    ana.tx->createBranch<vector<float>>("simhit_pz_pdgid_matched");
    ana.tx->createBranch<vector<float>>("ph2_x_pdgid_matched");
    ana.tx->createBranch<vector<float>>("ph2_y_pdgid_matched");
    ana.tx->createBranch<vector<float>>("ph2_z_pdgid_matched");


    for (unsigned int ilayer = 1; ilayer <= 3; ++ilayer)
    {
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_ncand", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_pass", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_passbits", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_algo", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_innerSg_innerMd_lower_hit_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_innerSg_innerMd_lower_hit_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_innerSg_innerMd_lower_hit_z", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_innerSg_innerMd_lower_hit_rt", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_innerSg_innerMd_lower_hit_side", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_innerSg_innerMd_lower_hit_rod", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_innerSg_innerMd_lower_hit_module", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_innerSg_innerMd_lower_hit_subdet", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_innerSg_innerMd_upper_hit_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_innerSg_innerMd_upper_hit_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_innerSg_innerMd_upper_hit_z", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_innerSg_innerMd_upper_hit_rt", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_innerSg_innerMd_upper_hit_side", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_innerSg_innerMd_upper_hit_rod", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_innerSg_innerMd_upper_hit_module", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_innerSg_innerMd_upper_hit_subdet", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_innerSg_outerMd_lower_hit_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_innerSg_outerMd_lower_hit_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_innerSg_outerMd_lower_hit_z", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_innerSg_outerMd_lower_hit_rt", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_innerSg_outerMd_lower_hit_side", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_innerSg_outerMd_lower_hit_rod", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_innerSg_outerMd_lower_hit_module", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_innerSg_outerMd_lower_hit_subdet", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_innerSg_outerMd_upper_hit_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_innerSg_outerMd_upper_hit_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_innerSg_outerMd_upper_hit_z", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_innerSg_outerMd_upper_hit_rt", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_innerSg_outerMd_upper_hit_side", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_innerSg_outerMd_upper_hit_rod", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_innerSg_outerMd_upper_hit_module", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_innerSg_outerMd_upper_hit_subdet", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_outerSg_innerMd_lower_hit_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_outerSg_innerMd_lower_hit_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_outerSg_innerMd_lower_hit_z", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_outerSg_innerMd_lower_hit_rt", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_outerSg_innerMd_lower_hit_side", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_outerSg_innerMd_lower_hit_rod", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_outerSg_innerMd_lower_hit_module", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_outerSg_innerMd_lower_hit_subdet", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_outerSg_innerMd_upper_hit_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_outerSg_innerMd_upper_hit_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_outerSg_innerMd_upper_hit_z", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_outerSg_innerMd_upper_hit_rt", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_outerSg_innerMd_upper_hit_side", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_outerSg_innerMd_upper_hit_rod", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_outerSg_innerMd_upper_hit_module", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_outerSg_innerMd_upper_hit_subdet", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_outerSg_outerMd_lower_hit_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_outerSg_outerMd_lower_hit_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_outerSg_outerMd_lower_hit_z", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_outerSg_outerMd_lower_hit_rt", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_outerSg_outerMd_lower_hit_side", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_outerSg_outerMd_lower_hit_rod", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_outerSg_outerMd_lower_hit_module", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_outerSg_outerMd_lower_hit_subdet", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_outerSg_outerMd_upper_hit_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_outerSg_outerMd_upper_hit_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_outerSg_outerMd_upper_hit_z", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_outerSg_outerMd_upper_hit_rt", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_outerSg_outerMd_upper_hit_side", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_outerSg_outerMd_upper_hit_rod", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_outerSg_outerMd_upper_hit_module", ilayer));
        ana.tx->createBranch<vector<int  >>(TString::Format("tl%d_outerSg_outerMd_upper_hit_subdet", ilayer));

        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaAv", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaAv_0th", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaAv_1st", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaAv_2nd", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaAv_3rd", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaAv_4th", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaIn", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaInRHmax", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaInRHmin", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaIn_cut", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaIn_0th", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaIn_1st", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaIn_1stCorr", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaIn_2nd", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaIn_2ndCorr", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaIn_3rd", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaIn_3rdCorr", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaIn_4th", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaOut", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaOutRHmax", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaOutRHmin", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaOut_0th", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaOut_1st", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaOut_1stCorr", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaOut_2nd", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaOut_2ndCorr", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaOut_3rd", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaOut_3rdCorr", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaOut_4th", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaPt_0th", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaPt_1st", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaPt_2nd", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaPt_3rd", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betaPt_4th", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_betacormode", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_dBeta", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_dBetaCut2", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_dBetaLum2", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_dBetaMuls", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_dBetaRIn2", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_dBetaROut2", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_dBetaRes", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_dBeta_0th", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_dBeta_1st", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_dBeta_2nd", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_dBeta_3rd", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_dBeta_4th", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_dBeta_midPoint", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_deltaZLum", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_dr", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_dzDrtScale", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_hit1_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_hit1_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_hit2_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_hit2_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_hit3_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_hit3_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_hit4_x", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_hit4_y", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_innerSgInnerMdDetId", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_innerSgOuterMdDetId", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_k2Rinv1GeVf", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_kRinv1GeVf", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_outerSgInnerMdDetId", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_outerSgOuterMdDetId", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_pixelPSZpitch", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_ptCut", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_pt_beta", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_pt_betaIn", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_pt_betaOut", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_rawBetaIn", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_rawBetaInCorrection", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_rawBetaOut", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_rawBetaOutCorrection", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_rtIn", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_rtOut", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_rtOut_o_rtIn", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_sdIn_alpha", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_sdIn_d", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_sdOut_alphaOut", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_sdOut_d", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_sdlSlope", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_sinAlphaMax", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_strip2SZpitch", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_zGeom", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_zIn", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_zLo", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_zHi", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_zLoPointed", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_zHiPointed", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_zOut", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_kZ", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_rtLo_point", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_rtHi_point", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_deltaPhiPos", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_dPhi", ilayer));
        ana.tx->createBranch<vector<float>>(TString::Format("tl%d_sdlCut", ilayer));

    }

}

void StudySDLTrackletDebugNtupleWriter::doStudy(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{

    ana.tx->clear();

    // Loop over track events
    for (auto& simtrkevent : simtrkevents)
    {
        // Unpack the tuple (sim_track_index, SDL::Event containing reco hits only matched to the given sim track)
        unsigned int& isimtrk = std::get<0>(simtrkevent);
        SDL::Event& trackevent = *(std::get<1>(simtrkevent));

        ana.tx->setBranch<float>("pt", trk.sim_pt()[isimtrk]);
        ana.tx->setBranch<float>("eta", trk.sim_eta()[isimtrk]);
        ana.tx->setBranch<float>("dxy", trk.sim_pca_dxy()[isimtrk]);
        ana.tx->setBranch<float>("dz", trk.sim_pca_dz()[isimtrk]);
        ana.tx->setBranch<float>("pdgid", trk.sim_pdgId()[isimtrk]);
        ana.tx->setBranch<float>("itrk", isimtrk);
        ana.tx->setBranch<int>("is_trk_bbbbbb", hasAll12HitsWithNBarrelUsingModuleMap(isimtrk, 6));
        ana.tx->setBranch<int>("is_trk_bbbbbe", hasAll12HitsWithNBarrelUsingModuleMap(isimtrk, 5));
        ana.tx->setBranch<int>("is_trk_bbbbee", hasAll12HitsWithNBarrelUsingModuleMap(isimtrk, 4));
        ana.tx->setBranch<int>("is_trk_bbbeee", hasAll12HitsWithNBarrelUsingModuleMap(isimtrk, 3));
        ana.tx->setBranch<int>("is_trk_bbeeee", hasAll12HitsWithNBarrelUsingModuleMap(isimtrk, 2));
        ana.tx->setBranch<int>("is_trk_beeeee", hasAll12HitsWithNBarrelUsingModuleMap(isimtrk, 1));


        // Writing out the sim hits
        for (unsigned int ith_hit = 0; ith_hit < trk.sim_simHitIdx()[isimtrk].size(); ++ith_hit)
        {

            // Retrieve the sim hit idx
            unsigned int simhitidx = trk.sim_simHitIdx()[isimtrk][ith_hit];

            // Select only the hits in the outer tracker
            if (not (trk.simhit_subdet()[simhitidx] == 4 or trk.simhit_subdet()[simhitidx] == 5))
                continue;

            // Select only the muon hits
            // if (not (trk.simhit_particle()[simhitidx] == trk.sim_pdgId()[isimtrk]))
            //     continue;

            // Exclude what I think is muon curling hit
            if (isMuonCurlingHit(isimtrk, ith_hit))
                break;

            ana.tx->pushbackToBranch<float>("simhit_x", trk.simhit_x()[simhitidx]);
            ana.tx->pushbackToBranch<float>("simhit_y", trk.simhit_y()[simhitidx]);
            ana.tx->pushbackToBranch<float>("simhit_z", trk.simhit_z()[simhitidx]);
            ana.tx->pushbackToBranch<float>("simhit_px", trk.simhit_px()[simhitidx]);
            ana.tx->pushbackToBranch<float>("simhit_py", trk.simhit_py()[simhitidx]);
            ana.tx->pushbackToBranch<float>("simhit_pz", trk.simhit_pz()[simhitidx]);
            for (unsigned int ireco_hit = 0; ireco_hit < trk.simhit_hitIdx()[simhitidx].size(); ++ireco_hit)
            {
                ana.tx->pushbackToBranch<float>("ph2_x", trk.ph2_x()[trk.simhit_hitIdx()[simhitidx][ireco_hit]]);
                ana.tx->pushbackToBranch<float>("ph2_y", trk.ph2_y()[trk.simhit_hitIdx()[simhitidx][ireco_hit]]);
                ana.tx->pushbackToBranch<float>("ph2_z", trk.ph2_z()[trk.simhit_hitIdx()[simhitidx][ireco_hit]]);
            }

            if (trk.simhit_particle()[simhitidx] == trk.sim_pdgId()[isimtrk])
            {
                ana.tx->pushbackToBranch<float>("simhit_x_pdgid_matched", trk.simhit_x()[simhitidx]);
                ana.tx->pushbackToBranch<float>("simhit_y_pdgid_matched", trk.simhit_y()[simhitidx]);
                ana.tx->pushbackToBranch<float>("simhit_z_pdgid_matched", trk.simhit_z()[simhitidx]);
                ana.tx->pushbackToBranch<float>("simhit_px_pdgid_matched", trk.simhit_px()[simhitidx]);
                ana.tx->pushbackToBranch<float>("simhit_py_pdgid_matched", trk.simhit_py()[simhitidx]);
                ana.tx->pushbackToBranch<float>("simhit_pz_pdgid_matched", trk.simhit_pz()[simhitidx]);
                for (unsigned int ireco_hit = 0; ireco_hit < trk.simhit_hitIdx()[simhitidx].size(); ++ireco_hit)
                {
                    ana.tx->pushbackToBranch<float>("ph2_x_pdgid_matched", trk.ph2_x()[trk.simhit_hitIdx()[simhitidx][ireco_hit]]);
                    ana.tx->pushbackToBranch<float>("ph2_y_pdgid_matched", trk.ph2_y()[trk.simhit_hitIdx()[simhitidx][ireco_hit]]);
                    ana.tx->pushbackToBranch<float>("ph2_z_pdgid_matched", trk.ph2_z()[trk.simhit_hitIdx()[simhitidx][ireco_hit]]);
                }
            }

        }

        for (unsigned int ilayer = 1; ilayer <= 3; ilayer++)
        {

            ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_ncand", ilayer), trackevent.getLayer(ilayer, SDL::Layer::Barrel).getTrackletPtrs().size() );
            for (auto& tlPtr : trackevent.getLayer(ilayer, SDL::Layer::Barrel).getTrackletPtrs())
            {

                tlPtr->runTrackletAlgo(SDL::Default_TLAlgo);
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_pass", ilayer), tlPtr->passesTrackletAlgo(SDL::Default_TLAlgo));
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_passbits", ilayer), tlPtr->getPassBitsDefaultAlgo());
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_algo", ilayer), tlPtr->getRecoVar("algo"));

                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_innerSg_innerMd_lower_hit_x"      , ilayer), tlPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->x()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_innerSg_innerMd_lower_hit_y"      , ilayer), tlPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->y()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_innerSg_innerMd_lower_hit_z"      , ilayer), tlPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->z()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_innerSg_innerMd_lower_hit_rt"     , ilayer), tlPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->rt()                );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_innerSg_innerMd_lower_hit_side"   , ilayer), tlPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().side()  );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_innerSg_innerMd_lower_hit_rod"    , ilayer), tlPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().rod()   );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_innerSg_innerMd_lower_hit_module" , ilayer), tlPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().module());
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_innerSg_innerMd_lower_hit_subdet" , ilayer), tlPtr->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet());
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_innerSg_innerMd_upper_hit_x"      , ilayer), tlPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->x()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_innerSg_innerMd_upper_hit_y"      , ilayer), tlPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->y()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_innerSg_innerMd_upper_hit_z"      , ilayer), tlPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->z()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_innerSg_innerMd_upper_hit_rt"     , ilayer), tlPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->rt()                );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_innerSg_innerMd_upper_hit_side"   , ilayer), tlPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->getModule().side()  );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_innerSg_innerMd_upper_hit_rod"    , ilayer), tlPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->getModule().rod()   );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_innerSg_innerMd_upper_hit_module" , ilayer), tlPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->getModule().module());
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_innerSg_innerMd_upper_hit_subdet" , ilayer), tlPtr->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->getModule().subdet());
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_innerSg_outerMd_lower_hit_x"      , ilayer), tlPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->x()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_innerSg_outerMd_lower_hit_y"      , ilayer), tlPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->y()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_innerSg_outerMd_lower_hit_z"      , ilayer), tlPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->z()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_innerSg_outerMd_lower_hit_rt"     , ilayer), tlPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->rt()                );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_innerSg_outerMd_lower_hit_side"   , ilayer), tlPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().side()  );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_innerSg_outerMd_lower_hit_rod"    , ilayer), tlPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().rod()   );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_innerSg_outerMd_lower_hit_module" , ilayer), tlPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().module());
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_innerSg_outerMd_lower_hit_subdet" , ilayer), tlPtr->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet());
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_innerSg_outerMd_upper_hit_x"      , ilayer), tlPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->x()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_innerSg_outerMd_upper_hit_y"      , ilayer), tlPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->y()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_innerSg_outerMd_upper_hit_z"      , ilayer), tlPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->z()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_innerSg_outerMd_upper_hit_rt"     , ilayer), tlPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->rt()                );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_innerSg_outerMd_upper_hit_side"   , ilayer), tlPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->getModule().side()  );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_innerSg_outerMd_upper_hit_rod"    , ilayer), tlPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->getModule().rod()   );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_innerSg_outerMd_upper_hit_module" , ilayer), tlPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->getModule().module());
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_innerSg_outerMd_upper_hit_subdet" , ilayer), tlPtr->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->getModule().subdet());
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_outerSg_innerMd_lower_hit_x"      , ilayer), tlPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->x()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_outerSg_innerMd_lower_hit_y"      , ilayer), tlPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->y()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_outerSg_innerMd_lower_hit_z"      , ilayer), tlPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->z()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_outerSg_innerMd_lower_hit_rt"     , ilayer), tlPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->rt()                );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_outerSg_innerMd_lower_hit_side"   , ilayer), tlPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().side()  );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_outerSg_innerMd_lower_hit_rod"    , ilayer), tlPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().rod()   );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_outerSg_innerMd_lower_hit_module" , ilayer), tlPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().module());
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_outerSg_innerMd_lower_hit_subdet" , ilayer), tlPtr->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet());
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_outerSg_innerMd_upper_hit_x"      , ilayer), tlPtr->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->x()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_outerSg_innerMd_upper_hit_y"      , ilayer), tlPtr->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->y()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_outerSg_innerMd_upper_hit_z"      , ilayer), tlPtr->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->z()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_outerSg_innerMd_upper_hit_rt"     , ilayer), tlPtr->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->rt()                );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_outerSg_innerMd_upper_hit_side"   , ilayer), tlPtr->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->getModule().side()  );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_outerSg_innerMd_upper_hit_rod"    , ilayer), tlPtr->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->getModule().rod()   );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_outerSg_innerMd_upper_hit_module" , ilayer), tlPtr->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->getModule().module());
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_outerSg_innerMd_upper_hit_subdet" , ilayer), tlPtr->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->getModule().subdet());
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_outerSg_outerMd_lower_hit_x"      , ilayer), tlPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->x()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_outerSg_outerMd_lower_hit_y"      , ilayer), tlPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->y()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_outerSg_outerMd_lower_hit_z"      , ilayer), tlPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->z()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_outerSg_outerMd_lower_hit_rt"     , ilayer), tlPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->rt()                );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_outerSg_outerMd_lower_hit_side"   , ilayer), tlPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().side()  );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_outerSg_outerMd_lower_hit_rod"    , ilayer), tlPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().rod()   );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_outerSg_outerMd_lower_hit_module" , ilayer), tlPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().module());
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_outerSg_outerMd_lower_hit_subdet" , ilayer), tlPtr->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->getModule().subdet());
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_outerSg_outerMd_upper_hit_x"      , ilayer), tlPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->x()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_outerSg_outerMd_upper_hit_y"      , ilayer), tlPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->y()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_outerSg_outerMd_upper_hit_z"      , ilayer), tlPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->z()                 );
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_outerSg_outerMd_upper_hit_rt"     , ilayer), tlPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->rt()                );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_outerSg_outerMd_upper_hit_side"   , ilayer), tlPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->getModule().side()  );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_outerSg_outerMd_upper_hit_rod"    , ilayer), tlPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->getModule().rod()   );
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_outerSg_outerMd_upper_hit_module" , ilayer), tlPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->getModule().module());
                ana.tx->pushbackToBranch<int  >(TString::Format("tl%d_outerSg_outerMd_upper_hit_subdet" , ilayer), tlPtr->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->getModule().subdet());

                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaAv", ilayer), tlPtr->getRecoVar("betaAv"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaAv_0th", ilayer), tlPtr->getRecoVar("betaAv_0th"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaAv_1st", ilayer), tlPtr->getRecoVar("betaAv_1st"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaAv_2nd", ilayer), tlPtr->getRecoVar("betaAv_2nd"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaAv_3rd", ilayer), tlPtr->getRecoVar("betaAv_3rd"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaAv_4th", ilayer), tlPtr->getRecoVar("betaAv_4th"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaIn", ilayer), tlPtr->getRecoVar("betaIn"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaInRHmax", ilayer), tlPtr->getRecoVar("betaInRHmax"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaInRHmin", ilayer), tlPtr->getRecoVar("betaInRHmin"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaIn_cut", ilayer), tlPtr->getRecoVar("betaIn_cut"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaIn_0th", ilayer), tlPtr->getRecoVar("betaIn_0th"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaIn_1st", ilayer), tlPtr->getRecoVar("betaIn_1st"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaIn_1stCorr", ilayer), tlPtr->getRecoVar("betaIn_1stCorr"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaIn_2nd", ilayer), tlPtr->getRecoVar("betaIn_2nd"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaIn_2ndCorr", ilayer), tlPtr->getRecoVar("betaIn_2ndCorr"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaIn_3rd", ilayer), tlPtr->getRecoVar("betaIn_3rd"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaIn_3rdCorr", ilayer), tlPtr->getRecoVar("betaIn_3rdCorr"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaIn_4th", ilayer), tlPtr->getRecoVar("betaIn_4th"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaOut", ilayer), tlPtr->getRecoVar("betaOut"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaOutRHmax", ilayer), tlPtr->getRecoVar("betaOutRHmax"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaOutRHmin", ilayer), tlPtr->getRecoVar("betaOutRHmin"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaOut_0th", ilayer), tlPtr->getRecoVar("betaOut_0th"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaOut_1st", ilayer), tlPtr->getRecoVar("betaOut_1st"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaOut_1stCorr", ilayer), tlPtr->getRecoVar("betaOut_1stCorr"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaOut_2nd", ilayer), tlPtr->getRecoVar("betaOut_2nd"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaOut_2ndCorr", ilayer), tlPtr->getRecoVar("betaOut_2ndCorr"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaOut_3rd", ilayer), tlPtr->getRecoVar("betaOut_3rd"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaOut_3rdCorr", ilayer), tlPtr->getRecoVar("betaOut_3rdCorr"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaOut_4th", ilayer), tlPtr->getRecoVar("betaOut_4th"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaPt_0th", ilayer), tlPtr->getRecoVar("betaPt_0th"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaPt_1st", ilayer), tlPtr->getRecoVar("betaPt_1st"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaPt_2nd", ilayer), tlPtr->getRecoVar("betaPt_2nd"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaPt_3rd", ilayer), tlPtr->getRecoVar("betaPt_3rd"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betaPt_4th", ilayer), tlPtr->getRecoVar("betaPt_4th"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_betacormode", ilayer), tlPtr->getRecoVar("betacormode"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_dBeta", ilayer), tlPtr->getRecoVar("dBeta"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_dBetaCut2", ilayer), tlPtr->getRecoVar("dBetaCut2"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_dBetaLum2", ilayer), tlPtr->getRecoVar("dBetaLum2"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_dBetaMuls", ilayer), tlPtr->getRecoVar("dBetaMuls"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_dBetaRIn2", ilayer), tlPtr->getRecoVar("dBetaRIn2"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_dBetaROut2", ilayer), tlPtr->getRecoVar("dBetaROut2"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_dBetaRes", ilayer), tlPtr->getRecoVar("dBetaRes"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_dBeta_0th", ilayer), tlPtr->getRecoVar("dBeta_0th"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_dBeta_1st", ilayer), tlPtr->getRecoVar("dBeta_1st"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_dBeta_2nd", ilayer), tlPtr->getRecoVar("dBeta_2nd"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_dBeta_3rd", ilayer), tlPtr->getRecoVar("dBeta_3rd"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_dBeta_4th", ilayer), tlPtr->getRecoVar("dBeta_4th"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_dBeta_midPoint", ilayer), tlPtr->getRecoVar("dBeta_midPoint"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_deltaZLum", ilayer), tlPtr->getRecoVar("deltaZLum"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_dr", ilayer), tlPtr->getRecoVar("dr"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_dzDrtScale", ilayer), tlPtr->getRecoVar("dzDrtScale"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_hit1_x", ilayer), tlPtr->getRecoVar("hit1_x"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_hit1_y", ilayer), tlPtr->getRecoVar("hit1_y"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_hit2_x", ilayer), tlPtr->getRecoVar("hit2_x"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_hit2_y", ilayer), tlPtr->getRecoVar("hit2_y"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_hit3_x", ilayer), tlPtr->getRecoVar("hit3_x"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_hit3_y", ilayer), tlPtr->getRecoVar("hit3_y"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_hit4_x", ilayer), tlPtr->getRecoVar("hit4_x"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_hit4_y", ilayer), tlPtr->getRecoVar("hit4_y"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_innerSgInnerMdDetId", ilayer), tlPtr->getRecoVar("innerSgInnerMdDetId"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_innerSgOuterMdDetId", ilayer), tlPtr->getRecoVar("innerSgOuterMdDetId"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_k2Rinv1GeVf", ilayer), tlPtr->getRecoVar("k2Rinv1GeVf"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_kRinv1GeVf", ilayer), tlPtr->getRecoVar("kRinv1GeVf"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_outerSgInnerMdDetId", ilayer), tlPtr->getRecoVar("outerSgInnerMdDetId"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_outerSgOuterMdDetId", ilayer), tlPtr->getRecoVar("outerSgOuterMdDetId"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_pixelPSZpitch", ilayer), tlPtr->getRecoVar("pixelPSZpitch"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_ptCut", ilayer), tlPtr->getRecoVar("ptCut"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_pt_beta", ilayer), tlPtr->getRecoVar("pt_beta"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_pt_betaIn", ilayer), tlPtr->getRecoVar("pt_betaIn"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_pt_betaOut", ilayer), tlPtr->getRecoVar("pt_betaOut"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_rawBetaIn", ilayer), tlPtr->getRecoVar("rawBetaIn"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_rawBetaInCorrection", ilayer), tlPtr->getRecoVar("rawBetaInCorrection"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_rawBetaOut", ilayer), tlPtr->getRecoVar("rawBetaOut"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_rawBetaOutCorrection", ilayer), tlPtr->getRecoVar("rawBetaOutCorrection"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_rtIn", ilayer), tlPtr->getRecoVar("rtIn"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_rtOut", ilayer), tlPtr->getRecoVar("rtOut"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_rtOut_o_rtIn", ilayer), tlPtr->getRecoVar("rtOut_o_rtIn"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_sdIn_alpha", ilayer), tlPtr->getRecoVar("sdIn_alpha"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_sdIn_d", ilayer), tlPtr->getRecoVar("sdIn_d"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_sdOut_alphaOut", ilayer), tlPtr->getRecoVar("sdOut_alphaOut"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_sdOut_d", ilayer), tlPtr->getRecoVar("sdOut_d"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_sdlSlope", ilayer), tlPtr->getRecoVar("sdlSlope"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_sinAlphaMax", ilayer), tlPtr->getRecoVar("sinAlphaMax"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_strip2SZpitch", ilayer), tlPtr->getRecoVar("strip2SZpitch"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_zGeom", ilayer), tlPtr->getRecoVar("zGeom"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_zIn", ilayer), tlPtr->getRecoVar("zIn"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_zLo", ilayer), tlPtr->getRecoVar("zLo"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_zHi", ilayer), tlPtr->getRecoVar("zHi"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_zLoPointed", ilayer), tlPtr->getRecoVar("zLoPointed"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_zHiPointed", ilayer), tlPtr->getRecoVar("zHiPointed"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_zOut", ilayer), tlPtr->getRecoVar("zOut"));

                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_kZ", ilayer), tlPtr->getRecoVar("kZ"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_rtLo_point", ilayer), tlPtr->getRecoVar("rtLo_point"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_rtHi_point", ilayer), tlPtr->getRecoVar("rtHi_point"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_deltaPhiPos", ilayer), tlPtr->getRecoVar("deltaPhiPos"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_dPhi", ilayer), tlPtr->getRecoVar("dPhi"));
                ana.tx->pushbackToBranch<float>(TString::Format("tl%d_sdlCut", ilayer), tlPtr->getRecoVar("sdlCut"));

            }


        }


        ana.tx->fill();
        ana.tx->clear();

    }

}

