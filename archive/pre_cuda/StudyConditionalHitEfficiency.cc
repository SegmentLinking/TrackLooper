#include "StudyConditionalHitEfficiency.h"

StudyConditionalHitEfficiency::StudyConditionalHitEfficiency(
        const char* studyName,
        std::vector<float> ptbounds,
        int pdgid
        )
{

    studyname = studyName;

    pt_boundaries = ptbounds;

    pdgid_of_study = pdgid;

}

void StudyConditionalHitEfficiency::bookStudy()
{
    // Book Histograms
    const int nlayers = NLAYERS;

    // std::vector<float> pt_boundaries_local = {
    //     0.5,
    //     0.52,
    //     0.54,
    //     0.56,
    //     0.58,
    //     0.6,
    //     0.62,
    //     0.64,
    //     0.66,
    //     0.68,
    //     0.7,
    //     0.72,
    //     0.74,
    //     0.76,
    //     0.78,
    //     0.8,
    //     0.82,
    //     0.84,
    //     0.86,
    //     0.88,
    //     0.9,
    //     0.92,
    //     0.94,
    //     0.96,
    //     0.98,
    //     1.0,
    //     1.02,
    //     1.04,
    //     1.06,
    //     1.08,
    //     1.1,
    //     1.12,
    //     1.14,
    //     1.16,
    //     1.18,
    //     1.2,
    //     1.22,
    //     1.24,
    //     1.26,
    //     1.28,
    //     1.3,
    //     1.32,
    //     1.34,
    //     1.36,
    //     1.38,
    //     1.4,
    //     1.42,
    //     1.44,
    //     1.46,
    //     1.48,
    //     1.5,
    //     1.52,
    //     1.54,
    //     1.56,
    //     1.58,
    //     1.6,
    //     1.62,
    //     1.64,
    //     1.66,
    //     1.68,
    //     1.7,
    //     1.72,
    //     1.74,
    //     1.76,
    //     1.78,
    //     1.8,
    //     1.82,
    //     1.84,
    //     1.86,
    //     1.88,
    //     1.9,
    //     1.92,
    //     1.94,
    //     1.96,
    //     1.98,
    //     2.0
    // };

    std::vector<float> pt_boundaries_local = {
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
        1.05,
        1.1,
        1.15,
        1.2,
        1.25,
        1.3,
        1.35,
        1.4,
        1.45,
        1.5,
        1.55,
        1.6,
        1.65,
        1.7,
        1.75,
        1.8,
        1.85,
        1.9,
        1.95,
        2.0
    };

    // std::vector<float> pt_boundaries_local = {
    //     1.0,
    //     5.0,
    //     10.0,
    //     15.0,
    //     20.0,
    //     25.0,
    //     30.0,
    //     35.0,
    //     40.0,
    //     45.0,
    //     50.0,
    // };

    ana.histograms.addVecHistogram("pt_all"  , pt_boundaries_local , [&]() { return pt_all;  } );
    ana.histograms.addVecHistogram("eta_all"  , 50, -1, 1, [&]() { return eta_all;  } );
    ana.histograms.addVecHistogram("phi_all"  , 1080, -3.1416, 3.1416, [&]() { return phi_all;  } );
    ana.histograms.addVecHistogram("pt_all_w_last_layer"  , pt_boundaries_local , [&]() { return pt_all_w_last_layer;  } );

    for (unsigned int i = 0; i < 13; ++i) ana.histograms.addVecHistogram(TString::Format("pt_w_simhit_miss%d", i)  , pt_boundaries_local , [&, i]() { return pt_w_nmiss_simhits[i];  } );
    for (unsigned int i = 0; i < 13; ++i) ana.histograms.addVecHistogram(TString::Format("pt_w_hit_miss%d", i)     , pt_boundaries_local , [&, i]() { return pt_w_nmiss_hits[i];     } );
    for (unsigned int i = 0; i < 13; ++i) ana.histograms.addVecHistogram(TString::Format("pt_w_miss_layer%d", i)   , pt_boundaries_local , [&, i]() { return pt_w_miss_layer[i];     } );
    for (unsigned int j = 0; j < 13; ++j) for (unsigned int i = 0; i < 13; ++i) ana.histograms.addVecHistogram(TString::Format("pt_w_nmiss%d_miss_layer%d", j, i)   , pt_boundaries_local , [&, i, j]() { return pt_w_nmiss_miss_layer[j][i];     } );

    for (unsigned int i = 0; i < 13; ++i) ana.histograms.addVecHistogram(TString::Format("eta_w_simhit_miss%d", i)  , 50, -1, 1, [&, i]() { return eta_w_nmiss_simhits[i];  } );
    for (unsigned int i = 0; i < 13; ++i) ana.histograms.addVecHistogram(TString::Format("eta_w_hit_miss%d", i)     , 50, -1, 1, [&, i]() { return eta_w_nmiss_hits[i];     } );
    for (unsigned int i = 0; i < 13; ++i) ana.histograms.addVecHistogram(TString::Format("eta_w_miss_layer%d", i)   , 50, -1, 1, [&, i]() { return eta_w_miss_layer[i];     } );
    for (unsigned int j = 0; j < 13; ++j) for (unsigned int i = 0; i < 13; ++i) ana.histograms.addVecHistogram(TString::Format("eta_w_nmiss%d_miss_layer%d", j, i)   , 50, -1, 1 , [&, i, j]() { return eta_w_nmiss_miss_layer[j][i];     } );

    for (unsigned int i = 0; i < 13; ++i) ana.histograms.addVecHistogram(TString::Format("phi_w_simhit_miss%d", i)  , 1080, -3.1416, 3.1416, [&, i]() { return phi_w_nmiss_simhits[i];  } );
    for (unsigned int i = 0; i < 13; ++i) ana.histograms.addVecHistogram(TString::Format("phi_w_hit_miss%d", i)     , 1080, -3.1416, 3.1416, [&, i]() { return phi_w_nmiss_hits[i];     } );
    for (unsigned int i = 0; i < 13; ++i) ana.histograms.addVecHistogram(TString::Format("phi_w_miss_layer%d", i)   , 1080, -3.1416, 3.1416, [&, i]() { return phi_w_miss_layer[i];     } );
    for (unsigned int j = 0; j < 13; ++j) for (unsigned int i = 0; i < 13; ++i) ana.histograms.addVecHistogram(TString::Format("phi_w_nmiss%d_miss_layer%d", j, i)   , 1080, -3.1416, 3.1416 , [&, i, j]() { return phi_w_nmiss_miss_layer[j][i];     } );

    ana.histograms.add2DVecHistogram("pt_1p5_1p75_nhits" , 12, 0, 12, "miss_layer" , 12, 0, 12, [&]() { return ptbin_nhits[0]; }, [&]() { return ptbin_miss_layer[0]; } );
    ana.histograms.add2DVecHistogram("pt_1p75_2p0_nhits" , 12, 0, 12, "miss_layer" , 12, 0, 12, [&]() { return ptbin_nhits[1]; }, [&]() { return ptbin_miss_layer[1]; } );

    for (unsigned int i = 0; i < 3; ++i) ana.histograms.addVecHistogram(TString::Format("prop_phi_2Slayer%d", i)  , 1080, -3.1416, 3.1416, [&, i]() { return prop_phi_2Slayer[i];  } );
    for (unsigned int i = 0; i < 3; ++i) ana.histograms.addVecHistogram(TString::Format("prop_phi_2Slayer%d_zoom_m02_p02", i)  , 1080, -0.2, 0.2, [&, i]() { return prop_phi_2Slayer[i];  } );
    for (unsigned int i = 0; i < 3; ++i) ana.histograms.addVecHistogram(TString::Format("prop_phi_2Slayer%d_zoom_m1_p1", i)  , 1080, -1.0, 1.0, [&, i]() { return prop_phi_2Slayer[i];  } );

    for (unsigned int i = 0; i < 3; ++i) ana.histograms.addVecHistogram(TString::Format("prop_phi_115_2Slayer%d", i)  , 1080, -3.1416, 3.1416, [&, i]() { return prop_115_phi_2Slayer[i];  } );
    for (unsigned int i = 0; i < 3; ++i) ana.histograms.addVecHistogram(TString::Format("prop_phi_115_2Slayer%d_zoom_m02_p02", i)  , 1080, -0.2, 0.2, [&, i]() { return prop_115_phi_2Slayer[i];  } );
    for (unsigned int i = 0; i < 3; ++i) ana.histograms.addVecHistogram(TString::Format("prop_phi_115_2Slayer%d_zoom_m1_p1", i)  , 1080, -1.0, 1.0, [&, i]() { return prop_115_phi_2Slayer[i];  } );

    ana.histograms.addVecHistogram("pt_0p95_1p05_hit_miss_study", 20, -1, 19, [&]() { return pt_0p95_1p05_hit_miss_study;  } );
    for (unsigned int i = 0; i < 3; ++i) ana.histograms.addVecHistogram(TString::Format("pt_0p95_1p05_nmiss2_prop_phi_2Slayer%d", i)  , 1080, -3.1416, 3.1416, [&, i]() { return pt_0p95_1p05_nmiss2_prop_phi_2Slayer[i];  } );
    for (unsigned int i = 0; i < 3; ++i) ana.histograms.addVecHistogram(TString::Format("pt_0p95_1p05_nmiss4_prop_phi_2Slayer%d_0", i)  , 1080, -3.1416, 3.1416, [&, i]() { return pt_0p95_1p05_nmiss4_prop_phi_2Slayer_0[i];  } );
    for (unsigned int i = 0; i < 3; ++i) ana.histograms.addVecHistogram(TString::Format("pt_0p95_1p05_nmiss4_prop_phi_2Slayer%d_1", i)  , 1080, -3.1416, 3.1416, [&, i]() { return pt_0p95_1p05_nmiss4_prop_phi_2Slayer_1[i];  } );

    // ana.tx->createBranch<float>("sim_pt");
    // ana.tx->createBranch<float>("sim_eta");
    // ana.tx->createBranch<float>("sim_phi");
    // ana.tx->createBranch<vector<float>>("simhit_x");
    // ana.tx->createBranch<vector<float>>("simhit_y");
    // ana.tx->createBranch<vector<float>>("simhit_z");
    // ana.tx->createBranch<vector<int  >>("simhit_side");
    // ana.tx->createBranch<vector<int  >>("simhit_rod");
    // ana.tx->createBranch<vector<int  >>("simhit_module");
    // ana.tx->createBranch<vector<int  >>("simhit_subdet");

}

float StudyConditionalHitEfficiency::prop_phi(unsigned int isimtrk, unsigned int originlayer, float targetR)
{
    for (auto& simhitidx : trk.sim_simHitIdx()[isimtrk])
    {

        if (trk.simhit_subdet()[simhitidx] != 5)
            continue;

        if (abs(trk.simhit_particle()[simhitidx]) != pdgid_of_study)
            continue;

        unsigned int detid = trk.simhit_detId()[simhitidx];
        SDL::Module module(detid);
        unsigned int layer = module.layer();
        unsigned int subdet = module.subdet();
        int islower = module.isLower();

        if (layer == originlayer and not islower)
        {
            TVector3 position;
            TVector3 momentum;
            position.SetXYZ(trk.simhit_x()[simhitidx], trk.simhit_y()[simhitidx], trk.simhit_z()[simhitidx]);
            momentum.SetXYZ(trk.simhit_px()[simhitidx], trk.simhit_py()[simhitidx], trk.simhit_pz()[simhitidx]);
            int pdgid = trk.simhit_particle()[simhitidx];
            int charge = 0;
            if (pdgid == 13)
                charge = -1;
            else
                charge = 1;

            if (charge > 0)
                continue;

            int status = -999;
            std::pair<TVector3, TVector3> result = helixPropagateApproxR(position,momentum,targetR,charge,status);
            return result.first.Phi();
        }
    }
}

void StudyConditionalHitEfficiency::doStudy(SDL::Event& event, std::vector<std::tuple<unsigned int, SDL::Event*>> simtrkevents)
{

    pt_all.clear();
    eta_all.clear();
    phi_all.clear();
    pt_all_w_last_layer.clear();

    // Reset values
    for (unsigned int i = 0; i < 13; ++i)
    {
        pt_w_nmiss_simhits[i].clear();
        pt_w_nmiss_hits[i].clear();
        pt_w_miss_layer[i].clear();
    }

    for (unsigned int i = 0; i < 13; ++i)
    {
        eta_w_nmiss_simhits[i].clear();
        eta_w_nmiss_hits[i].clear();
        eta_w_miss_layer[i].clear();
    }

    for (unsigned int i = 0; i < 13; ++i)
    {
        phi_w_nmiss_simhits[i].clear();
        phi_w_nmiss_hits[i].clear();
        phi_w_miss_layer[i].clear();
    }

    for (unsigned int j = 0; j < 13; ++j)
    {
        for (unsigned int i = 0; i < 13; ++i)
        {
            pt_w_nmiss_miss_layer[j][i].clear();
            eta_w_nmiss_miss_layer[j][i].clear();
            phi_w_nmiss_miss_layer[j][i].clear();
        }
    }

    for (unsigned int i = 0; i < 2; ++i)
    {
        ptbin_nhits[i].clear();
        ptbin_miss_layer[i].clear();
    }

    for (unsigned int i = 0; i < 3; ++i)
    {
        prop_phi_2Slayer[i].clear();
        prop_115_phi_2Slayer[i].clear();
    }

    pt_0p95_1p05_hit_miss_study.clear();

    for (unsigned int i = 0; i < 3; ++i)
    {
        pt_0p95_1p05_nmiss2_prop_phi_2Slayer[i].clear();
        pt_0p95_1p05_nmiss4_prop_phi_2Slayer_0[i].clear();
        pt_0p95_1p05_nmiss4_prop_phi_2Slayer_1[i].clear();
    }

    int n_denom_tracks = 0;

    // Loop over track events
    for (auto& simtrkevent : simtrkevents)
    {

        // Unpack the tuple (sim_track_index, SDL::Event containing reco hits only matched to the given sim track)
        unsigned int& isimtrk = std::get<0>(simtrkevent);
        SDL::Event& trackevent = *(std::get<1>(simtrkevent));

        // Parse pt and eta of this track
        float pt = std::min((double) trk.sim_pt()[isimtrk], 49.999);
        float eta = trk.sim_eta()[isimtrk];
        float phi = trk.sim_phi()[isimtrk];
        float dxy = trk.sim_pca_dxy()[isimtrk];
        float dz = trk.sim_pca_dz()[isimtrk];
        int id = abs(trk.sim_pdgId()[isimtrk]);

        if (abs(dxy) > 2.5)
            continue;

        if (abs(eta) > 0.8)
            continue;

        if (pt < 1.0)
            continue;

        // if (pt > 45)
        //     continue;

        if (id != 13)
            continue;

        if (abs(dz) > 10) 
            continue;

        pt_all.push_back(pt);
        if (pt > 0.7) eta_all.push_back(eta);

        ////=====================================================
        //// Require that we find a reco hit in the last layer

        //// First aggregate all the modules
        //std::vector<unsigned int> module_ids_in_last_layer;
        //for (auto& simhitidx : trk.sim_simHitIdx()[isimtrk])
        //{
        //    for (auto& recohitidx : trk.simhit_hitIdx()[simhitidx])
        //    {
        //        unsigned int detid = trk.ph2_detId()[recohitidx];
        //        SDL::Module module(detid);
        //        unsigned int layer = module.layer();
        //        unsigned int subdet = module.subdet();

        //        if (layer == 6 and subdet == 5)
        //        {
        //            module_ids_in_last_layer.push_back(detid);
        //        }
        //    }
        //}

        //// Check for a good mini-doublet pair
        //bool found_last_layer_good_minidoublet_candidate = false;
        //for (unsigned int ii = 0; ii < module_ids_in_last_layer.size(); ++ii)
        //{
        //    for (unsigned int jj = 1; jj < module_ids_in_last_layer.size(); ++jj)
        //    {
        //        if (SDL::Module(module_ids_in_last_layer[ii]).partnerDetId() == SDL::Module(module_ids_in_last_layer[jj]).detId())
        //            found_last_layer_good_minidoublet_candidate = true;
        //    }
        //}

        //if (not found_last_layer_good_minidoublet_candidate)
        //    continue;
        ////=====================================================

        ////=====================================================
        std::array<bool, 12> has_simhit_in_layer;
        has_simhit_in_layer.fill(0);

        for (auto& simhitidx : trk.sim_simHitIdx()[isimtrk])
        {

            if (trk.simhit_subdet()[simhitidx] != 5)
                continue;

            if (abs(trk.simhit_particle()[simhitidx]) != pdgid_of_study)
                continue;

            unsigned int detid = trk.simhit_detId()[simhitidx];
            SDL::Module module(detid);
            unsigned int layer = module.layer();
            unsigned int subdet = module.subdet();
            int islower = module.isLower();

            has_simhit_in_layer[(2*layer) - 1 - islower] = true;

        }

        int nmisssimhits = 0;
        for (unsigned int i = 0; i < 12; ++i)
        {
            if (not has_simhit_in_layer[i])
                nmisssimhits++;
        }

        // if (nmisssimhits == 12)
        // {

        //     // std::cout <<  " trk.sim_simHitIdx()[isimtrk].size(): " << trk.sim_simHitIdx()[isimtrk].size() <<  std::endl;
        //     int nsimhits = 0;
        //     for (auto& simhitidx : trk.sim_simHitIdx()[isimtrk])
        //     {

        //         if (trk.simhit_subdet()[simhitidx] != 5)
        //             continue;

        //         if (abs(trk.simhit_particle()[simhitidx]) != pdgid_of_study)
        //             continue;

        //         nsimhits++;

        //     }
        //     std::cout <<  " nsimhits: " << nsimhits <<  std::endl;
        // }

        pt_w_nmiss_simhits[nmisssimhits].push_back(pt);
        if (pt > 0.7) eta_w_nmiss_simhits[nmisssimhits].push_back(eta);
        if (pt > 1.0 and pt < 1.05) phi_w_nmiss_simhits[nmisssimhits].push_back(phi);

        for (unsigned int i = 0; i < 12; ++i)
        {
            if (not has_simhit_in_layer[i])
            {
                pt_w_nmiss_miss_layer[nmisssimhits][i].push_back(pt);
                if (pt > 0.7) eta_w_nmiss_miss_layer[nmisssimhits][i].push_back(eta);
                if (pt > 1.0 and pt < 1.05) phi_w_nmiss_miss_layer[nmisssimhits][i].push_back(phi);
            }
        }

        if (pt > 0.95 and pt < 1.05)
        {
            pt_0p95_1p05_hit_miss_study.push_back(-1); // denominator
            if (nmisssimhits < 0 or nmisssimhits > 12)
                std::cout <<  " nmisssimhits: " << nmisssimhits <<  std::endl;
            for (unsigned int i = 0; i < 13; ++i)
            {
                if (nmisssimhits == i)
                    pt_0p95_1p05_hit_miss_study.push_back(i); // numerators
            }

            if (nmisssimhits == 2)
            {
                int ilayer = -1;
                int jlayer = -1;
                for (unsigned int i = 0; i < 12; ++i)
                {
                    if (not has_simhit_in_layer[i])
                    {
                        if (ilayer < 0)
                            ilayer = i;
                        else
                            jlayer = i;
                    }
                }
                int nmiss_code = -1;
                if (ilayer == 6 and jlayer == 7)
                {
                    nmiss_code = 13;
                }
                if (ilayer == 8 and jlayer == 9)
                {
                    nmiss_code = 14;
                }
                if (ilayer == 10 and jlayer == 11)
                {
                    nmiss_code = 15;
                }

                pt_0p95_1p05_hit_miss_study.push_back(nmiss_code);

                float phipos = -999;
                if (nmiss_code == 13)
                {
                    phipos = prop_phi(isimtrk, 3, 68);
                    pt_0p95_1p05_nmiss2_prop_phi_2Slayer[0].push_back(phipos);
                }
                if (nmiss_code == 14)
                {
                    phipos = prop_phi(isimtrk, 4, 86);
                    pt_0p95_1p05_nmiss2_prop_phi_2Slayer[1].push_back(phipos);
                }
                if (nmiss_code == 15)
                {
                    phipos = prop_phi(isimtrk, 5, 110);
                    pt_0p95_1p05_nmiss2_prop_phi_2Slayer[2].push_back(phipos);
                }
            }
            if (nmisssimhits == 4)
            {
                int ilayer = -1;
                int jlayer = -1;
                int klayer = -1;
                int llayer = -1;
                for (unsigned int i = 0; i < 12; ++i)
                {
                    if (not has_simhit_in_layer[i])
                    {
                        if (ilayer < 0)
                            ilayer = i;
                        else if (jlayer < 0)
                            jlayer = i;
                        else if (klayer < 0)
                            klayer = i;
                        else if (llayer < 0)
                            llayer = i;
                    }
                }
                int nmiss_code = -1;
                if (ilayer == 6 and jlayer == 7 and klayer == 8 and llayer == 9)
                {
                    nmiss_code = 16;
                }
                if (ilayer == 6 and jlayer == 7 and klayer == 10 and llayer == 11)
                {
                    nmiss_code = 17;
                }
                if (ilayer == 8 and jlayer == 9 and klayer == 10 and llayer == 11)
                {
                    nmiss_code = 18;
                }

                pt_0p95_1p05_hit_miss_study.push_back(nmiss_code);

                float phipos = -999;
                if (nmiss_code == 16)
                {
                    phipos = prop_phi(isimtrk, 3, 68);
                    pt_0p95_1p05_nmiss4_prop_phi_2Slayer_0[0].push_back(phipos);
                    phipos = prop_phi(isimtrk, 3, 86);
                    pt_0p95_1p05_nmiss4_prop_phi_2Slayer_1[0].push_back(phipos);
                }
                if (nmiss_code == 17)
                {
                    phipos = prop_phi(isimtrk, 5, 68);
                    pt_0p95_1p05_nmiss4_prop_phi_2Slayer_0[1].push_back(phipos);
                    phipos = prop_phi(isimtrk, 5, 110);
                    pt_0p95_1p05_nmiss4_prop_phi_2Slayer_1[1].push_back(phipos);
                }
                if (nmiss_code == 18)
                {
                    phipos = prop_phi(isimtrk, 4, 86);
                    pt_0p95_1p05_nmiss4_prop_phi_2Slayer_0[2].push_back(phipos);
                    phipos = prop_phi(isimtrk, 4, 110);
                    pt_0p95_1p05_nmiss4_prop_phi_2Slayer_1[2].push_back(phipos);
                }
            }

        }

        if (nmisssimhits == 2)
        {

            // std::cout << std::endl;
            // std::cout << "[" << std::endl;

            std::vector<float> xs;
            std::vector<float> ys;

            for (auto& simhitidx : trk.sim_simHitIdx()[isimtrk])
            {

                if (trk.simhit_subdet()[simhitidx] != 5)
                    continue;

                if (abs(trk.simhit_particle()[simhitidx]) != pdgid_of_study)
                    continue;

                unsigned int detid = trk.simhit_detId()[simhitidx];
                SDL::Module module(detid);
                unsigned int layer = module.layer();
                unsigned int subdet = module.subdet();
                int islower = module.isLower();

                if (layer == 5 and not islower and (not has_simhit_in_layer[11]) and (not has_simhit_in_layer[10]))
                {
                    TVector3 position;
                    TVector3 momentum;
                    position.SetXYZ(trk.simhit_x()[simhitidx], trk.simhit_y()[simhitidx], trk.simhit_z()[simhitidx]);
                    momentum.SetXYZ(trk.simhit_px()[simhitidx], trk.simhit_py()[simhitidx], trk.simhit_pz()[simhitidx]);
                    int pdgid = trk.simhit_particle()[simhitidx];
                    int charge = 0;
                    if (pdgid == 13)
                        charge = -1;
                    else
                        charge = 1;

                    // if (charge > 0)
                    //     continue;

                    int status = -999;
                    std::pair<TVector3, TVector3> result = helixPropagateApproxR(position,momentum,110,charge,status);
                    prop_phi_2Slayer[2].push_back(result.first.Phi());
                    std::pair<TVector3, TVector3> result115 = helixPropagateApproxR(position,momentum,115,charge,status);
                    prop_115_phi_2Slayer[2].push_back(result115.first.Phi());
                    break;

                }

                if (layer == 4 and not islower and (not has_simhit_in_layer[9]) and (not has_simhit_in_layer[8]))
                {
                    TVector3 position;
                    TVector3 momentum;
                    position.SetXYZ(trk.simhit_x()[simhitidx], trk.simhit_y()[simhitidx], trk.simhit_z()[simhitidx]);
                    momentum.SetXYZ(trk.simhit_px()[simhitidx], trk.simhit_py()[simhitidx], trk.simhit_pz()[simhitidx]);
                    int pdgid = trk.simhit_particle()[simhitidx];
                    int charge = 0;
                    if (pdgid == 13)
                        charge = -1;
                    else
                        charge = 1;

                    // if (charge > 0)
                    //     continue;

                    int status = -999;
                    std::pair<TVector3, TVector3> result = helixPropagateApproxR(position,momentum,86,charge,status);
                    prop_phi_2Slayer[1].push_back(result.first.Phi());
                    break;

                }

                if (layer == 3 and not islower and (not has_simhit_in_layer[7]) and (not has_simhit_in_layer[6]))
                {
                    TVector3 position;
                    TVector3 momentum;
                    position.SetXYZ(trk.simhit_x()[simhitidx], trk.simhit_y()[simhitidx], trk.simhit_z()[simhitidx]);
                    momentum.SetXYZ(trk.simhit_px()[simhitidx], trk.simhit_py()[simhitidx], trk.simhit_pz()[simhitidx]);
                    int pdgid = trk.simhit_particle()[simhitidx];
                    int charge = 0;
                    if (pdgid == 13)
                        charge = -1;
                    else
                        charge = 1;

                    // if (charge > 0)
                    //     continue;

                    int status = -999;
                    std::pair<TVector3, TVector3> result = helixPropagateApproxR(position,momentum,68,charge,status);
                    prop_phi_2Slayer[0].push_back(result.first.Phi());
                    break;

                }

                float rt = sqrt(trk.simhit_x()[simhitidx]*trk.simhit_x()[simhitidx]+ trk.simhit_y()[simhitidx]*trk.simhit_y()[simhitidx]);
                // std::cout << trk.simhit_x()[simhitidx] << " " << trk.simhit_y()[simhitidx] << " " << rt << " " << trk.simhit_z()[simhitidx] << std::endl;

                xs.push_back(trk.simhit_x()[simhitidx]);
                ys.push_back(trk.simhit_y()[simhitidx]);

            }

            // std::cout << "[";
            // for (auto& i : xs)
            //     std::cout << i << ", ";
            // std::cout << "],";
            // std::cout << std::endl;
            // std::cout << "[";
            // for (auto& i : ys)
            //     std::cout << i << ", ";
            // std::cout << "],";
            // std::cout << std::endl;
            // std::cout << "],";
            // std::cout << std::endl;


        }


        if (nmisssimhits != 0)
            continue;
        ////=====================================================

        pt_all_w_last_layer.push_back(pt);

        // _______DENOM PASSED___________

        n_denom_tracks++;

        // Process which layers are missing and how many

        std::array<bool, 12> has_hit_in_layer;
        has_hit_in_layer.fill(0);

        for (auto& simhitidx : trk.sim_simHitIdx()[isimtrk])
        {

            if (trk.simhit_subdet()[simhitidx] != 5)
                continue;

            if (trk.simhit_particle()[simhitidx] != pdgid_of_study)
                continue;

            for (auto& recohitidx : trk.simhit_hitIdx()[simhitidx])
            {
                unsigned int detid = trk.ph2_detId()[recohitidx];
                SDL::Module module(detid);
                unsigned int layer = module.layer();
                unsigned int subdet = module.subdet();
                int islower = module.isLower();

                has_hit_in_layer[2 * layer - 1 - islower] = true;
            }
        }

        int nmisshits = 0;
        for (unsigned int i = 0; i < 12; ++i)
        {
            if (not has_hit_in_layer[i])
                nmisshits++;
        }
        // std::cout <<  " nmisshits: " << nmisshits <<  std::endl;
        pt_w_nmiss_hits[nmisshits].push_back(pt);
        if (pt > 0.7) eta_w_nmiss_hits[nmisshits].push_back(eta);

        if (nmisshits == 0)
        {
            pt_w_miss_layer[0].push_back(pt);
            if (pt > 0.7) eta_w_miss_layer[0].push_back(eta);
        }
        else
        {
            for (unsigned int i = 0; i < 12; ++i)
            {
                if (not has_hit_in_layer[i])
                {
                    pt_w_miss_layer[i+1].push_back(pt);
                    if (pt > 0.7) eta_w_miss_layer[i+1].push_back(eta);
                }
            }
        }

    }

    // std::cout <<  " n_denom_tracks: " << n_denom_tracks <<  std::endl;

}

