#include "trkCore.h"

//__________________________________________________________________________________________
float simhit_p(unsigned int simhitidx)
{

    // |momentum| calculation

    float px = trk.simhit_px()[simhitidx];
    float py = trk.simhit_py()[simhitidx];
    float pz = trk.simhit_pz()[simhitidx];
    return sqrt(px*px + py*py + pz*pz);
}

//__________________________________________________________________________________________
float hitAngle(unsigned int simhitidx)
{

    // This is the angle calculation between position vector and the momentum vector

    float x = trk.simhit_x()[simhitidx];
    float y = trk.simhit_y()[simhitidx];
    float z = trk.simhit_z()[simhitidx];
    float r3 = sqrt(x*x + y*y + z*z);
    float px = trk.simhit_px()[simhitidx];
    float py = trk.simhit_py()[simhitidx];
    float pz = trk.simhit_pz()[simhitidx];
    float p = sqrt(px*px + py*py + pz*pz);
    float rdotp = x*px + y*py + z*pz;
    rdotp = rdotp / r3;
    rdotp = rdotp / p;
    float angle = acos(rdotp);
    return angle;
}

//__________________________________________________________________________________________
bool isMuonCurlingHit(unsigned int isimtrk, unsigned int ith_hit)
{

    // To assess whether the ith_hit for isimtrk is a "curling" hit

    // Retrieve the sim hit idx
    unsigned int simhitidx = trk.sim_simHitIdx()[isimtrk][ith_hit];

    // We're only concerned about hits in the outer tracker
    // This is more of a sanity check
    if (not (trk.simhit_subdet()[simhitidx] == 4 or trk.simhit_subdet()[simhitidx] == 5))
        return false;

    // Retrieve the sim hit pdgId
    int simhit_particle = trk.simhit_particle()[simhitidx];

    // If the hit is not muon then we can't tell anything
    if (abs(simhit_particle) != 13)
        return false;

    // Get the angle
    float angle = hitAngle(simhitidx);

    // If the angle is quite different then it's the last hit
    if (abs(angle) > 1.6)
        return true;

    // Afterwards, we check the energy loss
    //
    // If it is the first hit without any previous hit present,
    // we can't tell whether it is last hit or not
    // so we just say false to be conservative
    if (ith_hit == 0)
        return false;

    // Retrieve the module where the hit lies
    int detid = trk.simhit_detId()[simhitidx];
    SDL::Module module = SDL::Module(detid);

    // Calculate the momentum
    float p = simhit_p(simhitidx);

    // Find the previous simhit that is on the lower side of the module
    int simhitidx_previous = -999;
    for (unsigned int ith_back = 1; ith_back <= ith_hit; ith_back++)
    {
        // Retrieve the hit idx of ith_hit - ith_back;
        unsigned int simhitidx_previous_candidate = trk.sim_simHitIdx()[isimtrk][ith_hit-ith_back];

        if (not (trk.simhit_subdet()[simhitidx_previous_candidate] == 4 or trk.simhit_subdet()[simhitidx_previous_candidate] == 5))
            continue;

        if (not (trk.simhit_particle()[simhitidx_previous_candidate] == 13))
            continue;

        // Retrieve the module where the previous candidate hit lies
        int detid = trk.simhit_detId()[simhitidx_previous_candidate];
        SDL::Module module = SDL::Module(detid);

        // If the module is lower, then we get the index
        if (module.isLower())
        {
            simhitidx_previous = simhitidx_previous_candidate;
            break;
        }

    }

    // If no previous layer is found then can't do much
    if (simhitidx_previous == -999)
        return false;

    // Get the previous layer momentum
    float p_previous = simhit_p(simhitidx_previous);

    // Calculate the momentum loss
    float loss = fabs(p_previous - p) / p_previous;

    // If the momentum loss is large it is the last hit
    if (loss > 0.35)
        return true;

    // If it reaches this point again, we're not sure what this hit is
    // So we return false 
    return false;

}

//__________________________________________________________________________________________
bool hasAll12HitsWithNBarrelUsingModuleMap(unsigned int isimtrk, int nbarrel, bool usesimhits)
{

    // Select only tracks that left all 12 hits in the barrel
    std::array<std::vector<SDL::Module>, 6> layers_modules_barrel; // Watch out for duplicates in this vector, do not count with this for unique count.
    std::array<std::vector<SDL::Module>, 6> layers_modules_endcap; // Watch out for duplicates in this vector, do not count with this for unique count.

    std::vector<float> ps;

    for (unsigned int ith_hit = 0; ith_hit < trk.sim_simHitIdx()[isimtrk].size(); ++ith_hit)
    {

        // Retrieve the sim hit idx
        unsigned int simhitidx = trk.sim_simHitIdx()[isimtrk][ith_hit];

        // Select only the hits in the outer tracker
        if (not (trk.simhit_subdet()[simhitidx] == 4 or trk.simhit_subdet()[simhitidx] == 5))
            continue;

        if (not (trk.simhit_particle()[simhitidx] == trk.sim_pdgId()[isimtrk]))
            continue;

        if (isMuonCurlingHit(isimtrk, ith_hit))
            break;

        if (not usesimhits)
        {

            // list of reco hit matched to this sim hit
            for (unsigned int irecohit = 0; irecohit < trk.simhit_hitIdx()[simhitidx].size(); ++irecohit)
            {
                // Get the recohit type
                int recohittype = trk.simhit_hitType()[simhitidx][irecohit];

                // Consider only ph2 hits (i.e. outer tracker hits)
                if (recohittype == 4)
                {

                    int ihit = trk.simhit_hitIdx()[simhitidx][irecohit];

                    if (trk.ph2_subdet()[ihit] == 5)
                    {
                        layers_modules_barrel[trk.ph2_layer()[ihit] - 1].push_back(SDL::Module(trk.ph2_detId()[ihit]));
                    }
                    if (trk.ph2_subdet()[ihit] == 4)
                    {
                        layers_modules_endcap[trk.ph2_layer()[ihit] - 1].push_back(SDL::Module(trk.ph2_detId()[ihit]));
                    }

                }

            }

        }
        else
        {

            if (trk.simhit_hitIdx()[simhitidx].size() > 0)
            {

                if (trk.simhit_subdet()[simhitidx] == 5)
                {
                    layers_modules_barrel[trk.simhit_layer()[simhitidx] - 1].push_back(SDL::Module(trk.simhit_detId()[simhitidx]));
                }
                if (trk.simhit_subdet()[simhitidx] == 4)
                {
                    layers_modules_endcap[trk.simhit_layer()[simhitidx] - 1].push_back(SDL::Module(trk.simhit_detId()[simhitidx]));
                }
            }

            // // list of reco hit matched to this sim hit
            // for (unsigned int irecohit = 0; irecohit < trk.simhit_hitIdx()[simhitidx].size(); ++irecohit)
            // {
            //     // Get the recohit type
            //     int recohittype = trk.simhit_hitType()[simhitidx][irecohit];

            //     // Consider only ph2 hits (i.e. outer tracker hits)
            //     if (recohittype == 4)
            //     {

            //         int ihit = trk.simhit_hitIdx()[simhitidx][irecohit];

            //         if (trk.ph2_subdet()[ihit] == 5)
            //         {
            //             layers_modules_barrel[trk.ph2_layer()[ihit] - 1].push_back(SDL::Module(trk.ph2_detId()[ihit]));
            //         }
            //         if (trk.ph2_subdet()[ihit] == 4)
            //         {
            //             layers_modules_endcap[trk.ph2_layer()[ihit] - 1].push_back(SDL::Module(trk.ph2_detId()[ihit]));
            //         }

            //     }

            // }

        }

    }

    // Aggregating good pair modules (i.e. a double module with each side having non-zero reco hits.)
    std::array<std::vector<unsigned int>, 6> layers_good_paired_modules; // Watch out for duplicates in this vector, do not count with this for unique count.

    for (int logical_ilayer = 0; logical_ilayer < 6; ++logical_ilayer)
    {
        // Raw layer number
        bool is_ilayer_barrel = logical_ilayer < nbarrel;
        int raw_ilayer = is_ilayer_barrel ? logical_ilayer: logical_ilayer - nbarrel;
        const std::array<std::vector<SDL::Module>, 6>& layers_modules = is_ilayer_barrel ? layers_modules_barrel : layers_modules_endcap;

        // Then get the module in the ilayer and check that it has a good module pair
        // Loop over modules in the given raw_ilayer and match the pairs and if a good pair of modules have hits in each module
        // then save the lower modules to layers_good_paired_modules.
        // NOTE there may be duplicates being pushed to layers_good_paired_modules
        // Do not count with these vectors
        for (unsigned imod = 0; imod < layers_modules[raw_ilayer].size(); ++imod)
        {
            for (unsigned jmod = imod + 1; jmod < layers_modules[raw_ilayer].size(); ++jmod)
            {
                // if two modules are a good pair
                if (layers_modules[raw_ilayer][imod].partnerDetId() == layers_modules[raw_ilayer][jmod].detId())
                {
                    // add the lower module one to the good_paired_modules list
                    if (layers_modules[raw_ilayer][imod].isLower())
                    {
                        if (std::find(
                                    layers_good_paired_modules[logical_ilayer].begin(),
                                    layers_good_paired_modules[logical_ilayer].end(),
                                    layers_modules[raw_ilayer][imod].detId()) == layers_good_paired_modules[logical_ilayer].end())
                            layers_good_paired_modules[logical_ilayer].push_back(layers_modules[raw_ilayer][imod].detId());
                    }
                    else
                    {
                        if (std::find(
                                    layers_good_paired_modules[logical_ilayer].begin(),
                                    layers_good_paired_modules[logical_ilayer].end(),
                                    layers_modules[raw_ilayer][imod].partnerDetId()) == layers_good_paired_modules[logical_ilayer].end())
                        layers_good_paired_modules[logical_ilayer].push_back(layers_modules[raw_ilayer][imod].partnerDetId());
                    }
                }
            }
        }
    }

    return checkModuleConnectionsAreGood(layers_good_paired_modules);

}

//__________________________________________________________________________________________
bool checkModuleConnectionsAreGood(std::array<std::vector<unsigned int>, 6>& layers_good_paired_modules)
{
    // Dumbest possible solution
    for (auto& module0 : layers_good_paired_modules[0])
    {
        const std::vector<unsigned int>& connectedModule1s = ana.moduleConnectiongMapLoose.getConnectedModuleDetIds(module0);
        for (auto& module1 : layers_good_paired_modules[1])
        {
            if (std::find(connectedModule1s.begin(), connectedModule1s.end(), module1) == connectedModule1s.end())
                break;
            const std::vector<unsigned int>& connectedModule2s = ana.moduleConnectiongMapLoose.getConnectedModuleDetIds(module1);
            for (auto& module2 : layers_good_paired_modules[2])
            {
                if (std::find(connectedModule2s.begin(), connectedModule2s.end(), module2) == connectedModule2s.end())
                    break;
                const std::vector<unsigned int>& connectedModule3s = ana.moduleConnectiongMapLoose.getConnectedModuleDetIds(module2);
                for (auto& module3 : layers_good_paired_modules[3])
                {
                    if (std::find(connectedModule3s.begin(), connectedModule3s.end(), module3) == connectedModule3s.end())
                        break;
                    const std::vector<unsigned int>& connectedModule4s = ana.moduleConnectiongMapLoose.getConnectedModuleDetIds(module3);
                    for (auto& module4 : layers_good_paired_modules[4])
                    {
                        if (std::find(connectedModule4s.begin(), connectedModule4s.end(), module4) == connectedModule4s.end())
                            break;
                        const std::vector<unsigned int>& connectedModule5s = ana.moduleConnectiongMapLoose.getConnectedModuleDetIds(module4);
                        for (auto& module5 : layers_good_paired_modules[5])
                        {
                            if (std::find(connectedModule5s.begin(), connectedModule5s.end(), module5) == connectedModule5s.end())
                                break;
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}

//__________________________________________________________________________________________
bool hasAll12HitsWithNBarrel(unsigned int isimtrk, int nbarrel)
{

    // Select only tracks that left all 12 hits in the barrel
    std::array<std::vector<SDL::Module>, 6> layers_modules_barrel;
    std::array<std::vector<SDL::Module>, 6> layers_modules_endcap;

    std::vector<float> ps;

    for (unsigned int ith_hit = 0; ith_hit < trk.sim_simHitIdx()[isimtrk].size(); ++ith_hit)
    {

        // Retrieve the sim hit idx
        unsigned int simhitidx = trk.sim_simHitIdx()[isimtrk][ith_hit];

        // Select only the hits in the outer tracker
        if (not (trk.simhit_subdet()[simhitidx] == 4 or trk.simhit_subdet()[simhitidx] == 5))
            continue;

        // if (not (trk.simhit_particle()[simhitidx] == trk.sim_pdgId()[isimtrk]))
        //     continue;

        if (isMuonCurlingHit(isimtrk, ith_hit))
            break;

        // list of reco hit matched to this sim hit
        for (unsigned int irecohit = 0; irecohit < trk.simhit_hitIdx()[simhitidx].size(); ++irecohit)
        {
            // Get the recohit type
            int recohittype = trk.simhit_hitType()[simhitidx][irecohit];

            // Consider only ph2 hits (i.e. outer tracker hits)
            if (recohittype == 4)
            {

                int ihit = trk.simhit_hitIdx()[simhitidx][irecohit];

                if (trk.ph2_subdet()[ihit] == 5)
                {
                    layers_modules_barrel[trk.ph2_layer()[ihit] - 1].push_back(SDL::Module(trk.ph2_detId()[ihit]));
                }
                if (trk.ph2_subdet()[ihit] == 4)
                {
                    layers_modules_endcap[trk.ph2_layer()[ihit] - 1].push_back(SDL::Module(trk.ph2_detId()[ihit]));
                }

            }

        }

    }

    std::array<bool, 6> has_good_pair_by_layer_barrel;
    has_good_pair_by_layer_barrel[0] = false;
    has_good_pair_by_layer_barrel[1] = false;
    has_good_pair_by_layer_barrel[2] = false;
    has_good_pair_by_layer_barrel[3] = false;
    has_good_pair_by_layer_barrel[4] = false;
    has_good_pair_by_layer_barrel[5] = false;

    std::array<bool, 6> has_good_pair_by_layer_endcap;
    has_good_pair_by_layer_endcap[0] = false;
    has_good_pair_by_layer_endcap[1] = false;
    has_good_pair_by_layer_endcap[2] = false;
    has_good_pair_by_layer_endcap[3] = false;
    has_good_pair_by_layer_endcap[4] = false;
    has_good_pair_by_layer_endcap[5] = false;

    bool has_good_pair_all_layer = true;

    for (int ilayer = 0; ilayer < 6; ++ilayer)
    {

        bool has_good_pair = false;

        if (ilayer < nbarrel)
        {

            for (unsigned imod = 0; imod < layers_modules_barrel[ilayer].size(); ++imod)
            {
                for (unsigned jmod = imod + 1; jmod < layers_modules_barrel[ilayer].size(); ++jmod)
                {
                    if (layers_modules_barrel[ilayer][imod].partnerDetId() == layers_modules_barrel[ilayer][jmod].detId())
                        has_good_pair = true;
                }
            }

        }
        else
        {

            int endcap_ilayer = ilayer - nbarrel;

            for (unsigned imod = 0; imod < layers_modules_endcap[endcap_ilayer].size(); ++imod)
            {
                for (unsigned jmod = imod + 1; jmod < layers_modules_endcap[endcap_ilayer].size(); ++jmod)
                {
                    if (layers_modules_endcap[endcap_ilayer][imod].partnerDetId() == layers_modules_endcap[endcap_ilayer][jmod].detId())
                        has_good_pair = true;
                }
            }

        }

        if (not has_good_pair)
            has_good_pair_all_layer = false;

        if (ilayer < nbarrel)
            has_good_pair_by_layer_barrel[ilayer] = has_good_pair;
        else
            has_good_pair_by_layer_endcap[ilayer] = has_good_pair;

    }


    float pt = trk.sim_pt()[isimtrk];
    float eta = trk.sim_eta()[isimtrk];

    // if (abs((trk.sim_pt()[isimtrk] - 0.71710)) < 0.00001)
    // {
    //     std::cout << std::endl;
    //     std::cout <<  " has_good_pair_by_layer[0]: " << has_good_pair_by_layer[0] <<  " has_good_pair_by_layer[1]: " << has_good_pair_by_layer[1] <<  " has_good_pair_by_layer[2]: " << has_good_pair_by_layer[2] <<  " has_good_pair_by_layer[3]: " << has_good_pair_by_layer[3] <<  " has_good_pair_by_layer[4]: " << has_good_pair_by_layer[4] <<  " has_good_pair_by_layer[5]: " << has_good_pair_by_layer[5] <<  " pt: " << pt <<  " eta: " << eta <<  std::endl;
    // }

    return has_good_pair_all_layer;

}

//__________________________________________________________________________________________
// Check for at least one sim hit in each layer and nothing more
bool goodBarrelTrack(unsigned int isimtrk, int pdgid)
{

    std::vector<int> layers;

    for (unsigned int ith_hit = 0; ith_hit < trk.sim_simHitIdx()[isimtrk].size(); ++ith_hit)
    {

        // Retrieve the sim hit idx
        unsigned int simhitidx = trk.sim_simHitIdx()[isimtrk][ith_hit];

        // Select only the hits in barrel
        if (not (trk.simhit_subdet()[simhitidx] == 5))
            continue;

        // Select only sim hits matching the particle pdgid
        if (not (trk.simhit_particle()[simhitidx] == trk.sim_pdgId()[isimtrk]))
            continue;

        // if pdgid is provided then check that the pdgid 
        if (pdgid != 0)
            if (not (trk.sim_pdgId()[isimtrk] == abs(pdgid)))
                continue;

        // add to layers
        layers.push_back(trk.simhit_layer()[simhitidx]);

    }

    if (not (std::find(layers.begin(), layers.end(), 1) != layers.end())) return false;
    if (not (std::find(layers.begin(), layers.end(), 2) != layers.end())) return false;
    if (not (std::find(layers.begin(), layers.end(), 3) != layers.end())) return false;
    if (not (std::find(layers.begin(), layers.end(), 4) != layers.end())) return false;
    if (not (std::find(layers.begin(), layers.end(), 5) != layers.end())) return false;
    if (not (std::find(layers.begin(), layers.end(), 6) != layers.end())) return false;
    return true;

}

//__________________________________________________________________________________________
bool hasAll12HitsInBarrel(unsigned int isimtrk)
{

    // Select only tracks that left all 12 hits in the barrel
    std::array<std::vector<SDL::Module>, 6> layers_modules;

    std::vector<float> ps;

    for (unsigned int ith_hit = 0; ith_hit < trk.sim_simHitIdx()[isimtrk].size(); ++ith_hit)
    {

        // Retrieve the sim hit idx
        unsigned int simhitidx = trk.sim_simHitIdx()[isimtrk][ith_hit];

        // Select only the hits in the outer tracker
        if (not (trk.simhit_subdet()[simhitidx] == 4 or trk.simhit_subdet()[simhitidx] == 5))
            continue;

        // if (not (trk.simhit_particle()[simhitidx] == trk.sim_pdgId()[isimtrk]))
        //     continue;

        if (isMuonCurlingHit(isimtrk, ith_hit))
            break;

        // list of reco hit matched to this sim hit
        for (unsigned int irecohit = 0; irecohit < trk.simhit_hitIdx()[simhitidx].size(); ++irecohit)
        {
            // Get the recohit type
            int recohittype = trk.simhit_hitType()[simhitidx][irecohit];

            // Consider only ph2 hits (i.e. outer tracker hits)
            if (recohittype == 4)
            {

                int ihit = trk.simhit_hitIdx()[simhitidx][irecohit];

                if (trk.ph2_subdet()[ihit] == 5)
                {
                    layers_modules[trk.ph2_layer()[ihit] - 1].push_back(SDL::Module(trk.ph2_detId()[ihit]));
                }

            }

        }

    }

    std::array<bool, 6> has_good_pair_by_layer;
    has_good_pair_by_layer[0] = false;
    has_good_pair_by_layer[1] = false;
    has_good_pair_by_layer[2] = false;
    has_good_pair_by_layer[3] = false;
    has_good_pair_by_layer[4] = false;
    has_good_pair_by_layer[5] = false;

    bool has_good_pair_all_layer = true;

    for (int ilayer = 0; ilayer < 6; ++ilayer)
    {

        bool has_good_pair = false;

        for (unsigned imod = 0; imod < layers_modules[ilayer].size(); ++imod)
        {
            for (unsigned jmod = imod + 1; jmod < layers_modules[ilayer].size(); ++jmod)
            {
                if (layers_modules[ilayer][imod].partnerDetId() == layers_modules[ilayer][jmod].detId())
                    has_good_pair = true;
            }
        }

        if (not has_good_pair)
            has_good_pair_all_layer = false;

        has_good_pair_by_layer[ilayer] = has_good_pair;

    }

    float pt = trk.sim_pt()[isimtrk];
    float eta = trk.sim_eta()[isimtrk];

    // if (abs((trk.sim_pt()[isimtrk] - 0.71710)) < 0.00001)
    // {
    //     std::cout << std::endl;
    //     std::cout <<  " has_good_pair_by_layer[0]: " << has_good_pair_by_layer[0] <<  " has_good_pair_by_layer[1]: " << has_good_pair_by_layer[1] <<  " has_good_pair_by_layer[2]: " << has_good_pair_by_layer[2] <<  " has_good_pair_by_layer[3]: " << has_good_pair_by_layer[3] <<  " has_good_pair_by_layer[4]: " << has_good_pair_by_layer[4] <<  " has_good_pair_by_layer[5]: " << has_good_pair_by_layer[5] <<  " pt: " << pt <<  " eta: " << eta <<  std::endl;
    // }

    return has_good_pair_all_layer;

}



bool hasAtLeastOneHitPairinEndcapLikeTiltedModule(unsigned short layer, unsigned int isimtrk)
{
    //Checking done only for the layer specified by "layer", otherwise function returns true always
    std::vector<SDL::Module> layer_modules;


    for (unsigned int ith_hit = 0; ith_hit < trk.sim_simHitIdx()[isimtrk].size(); ++ith_hit)
    {

        // Retrieve the sim hit idx
        unsigned int simhitidx = trk.sim_simHitIdx()[isimtrk][ith_hit];

        // Select only the hits in the outer tracker
        if (not (trk.simhit_subdet()[simhitidx] == 4 or trk.simhit_subdet()[simhitidx] == 5))
            continue;

        if (isMuonCurlingHit(isimtrk, ith_hit))
            break;

        // list of reco hit matched to this sim hit
        for (unsigned int irecohit = 0; irecohit < trk.simhit_hitIdx()[simhitidx].size(); ++irecohit)
        {
            // Get the recohit type
            int recohittype = trk.simhit_hitType()[simhitidx][irecohit];

            // Consider only ph2 hits (i.e. outer tracker hits)
            if (recohittype == 4)
            {

                int ihit = trk.simhit_hitIdx()[simhitidx][irecohit];


                if(trk.ph2_layer()[ihit] == layer and trk.ph2_subdet()[ihit] == 5 and (not SDL::MiniDoublet::useBarrelLogic(trk.ph2_detId()[ihit])))
                {
                    layer_modules.push_back(SDL::Module(trk.ph2_detId()[ihit]));

                }

            }

        }
    }
    //Check if there is at least one pair in an endcap like tilted module
    //in the layer denoted by the parameter "layer"

    bool hasTiltedPairAtLayer = false;

    for (unsigned imod = 0; imod < layer_modules.size(); ++imod)
    {

        for (unsigned jmod = imod + 1; jmod < layer_modules.size(); ++jmod)
        {
            if (layer_modules[imod].partnerDetId() == layer_modules[jmod].detId())
            {
                hasTiltedPairAtLayer = true;
            }
                
        }
    }

    return hasTiltedPairAtLayer;
}



//__________________________________________________________________________________________
bool isMTVMatch(unsigned int isimtrk, std::vector<unsigned int> hit_idxs)
{
    std::vector<unsigned int> sim_trk_ihits;
    for (auto& i_simhit_idx : trk.sim_simHitIdx()[isimtrk])
    {
        for (auto& ihit : trk.simhit_hitIdx()[i_simhit_idx])
        {
            sim_trk_ihits.push_back(ihit);
        }
    }

    std::vector<unsigned int> v_intersection;
 
    std::set_intersection(sim_trk_ihits.begin(), sim_trk_ihits.end(),
                          hit_idxs.begin(), hit_idxs.end(),
                          std::back_inserter(v_intersection));

    // If 75% of 12 hits have been found than it is matched
    return (v_intersection.size() > 9);
}

//__________________________________________________________________________________________
bool is75percentFromSimMatchedHits(std::vector<unsigned int> hitidxs, int pdgid)
{
    std::vector<unsigned int> hitidxs_w_matched_pdgid;
    for (auto& i_reco_hit : hitidxs)
    {
        bool matched = false;
        for (auto& i_simhit_idx : trk.ph2_simHitIdx()[i_reco_hit])
        {
            matched = true;
            break;
            // int particle_id = trk.simhit_particle()[i_simhit_idx];
            // if (abs(pdgid) == abs(particle_id))
            // {
            //     matched = true;
            //     break;
            // }
        }
        if (matched)
            hitidxs_w_matched_pdgid.push_back(i_reco_hit);
    }

    std::vector<unsigned int> v_intersection;
 
    std::set_intersection(hitidxs_w_matched_pdgid.begin(), hitidxs_w_matched_pdgid.end(),
                          hitidxs.begin(), hitidxs.end(),
                          std::back_inserter(v_intersection));

    // If 75% of 12 hits have been found than it is matched
    return (v_intersection.size() >= 9);

}

//__________________________________________________________________________________________
TVector3 linePropagateR(const TVector3& r3, const TVector3& p3, double rDest, int& status, bool useClosest, bool verbose)
{
  double rt = r3.Pt();
  double d = rDest - rt;

  double dotPR2D = r3.x()*p3.x() + r3.y()*p3.y();

  double pt = p3.Pt();
  double p =  p3.Mag();
  
  // r3 + p3/p*x*|d| = dest : dest.t = rt + d <=> rt^2 + 2*dotPR2D/p*x*|d| + pt^2/p^2*x^2*d^2 = rt^2 + 2*rt*d + d^2
  // 2*dotPR2D/p*|d|* x + pt^2/p^2*d^2* x^2 - ( 2*rt*d + d^2) = 0
  // 2*dotPR2D/p/|d|* x + pt^2/p^2* x^2 - ( 2*rt/d + 1) = 0
  // x^2 + 2*dotPR2D/p/|d|*(p/pt)^2* x  - ( 2*rt/d + 1)*(p/pt)^2 = 0
  // - dotPR2D/p/|d|*(p/pt)^2  +/- sqrt( (dotPR2D/p/|d|*(p/pt)^2)^2 + ( 2*rt/d + 1)*(p/pt)^2 )
  // (p/pt)*( - dotPR2D/pt/|d|  +/- sqrt( (dotPR2D/pt/|d| )^2 + ( 2*rt/d + 1) ) )
  double bb = dotPR2D/pt/std::abs(d);
  double disc = bb*bb + (2.*rt/d + 1.);
  status = 0;
  if (disc < 0){
    status = 1;
    return r3;
  }
  double dSign = useClosest ? 1. : -1.;
  double xxP = (p/pt)*( sqrt(bb*bb + (2.*rt/d + 1.)) - bb);
  double xxM = (p/pt)*( - sqrt(bb*bb + (2.*rt/d + 1.)) - bb);
  double xx;
  if (useClosest){
    xx = std::abs(xxP) < std::abs(xxM) ? xxP : xxM;
  } else {
    xx = std::abs(xxP) < std::abs(xxM) ? xxM : xxP;
  }
  TVector3 dest = r3 + p3*(std::abs(d)/p)*xx;
  if (verbose || std::abs(dest.Pt() - rDest)>0.001){
    std::cout<<"linePropagateR "<<r3.Pt()<<" "<<r3.Phi()<<" "<<r3.z()<<" "<<pt<<" "<<p
	     <<" "<<d<<" "<<r3.x()*p3.x()<<" "<<r3.y()*p3.y()<<" "<<dotPR2D<<" "<<bb<<" "<<(2.*rt/d + 1.)<<" "<<bb*bb + (2.*rt/d + 1.)
	     <<" => "<<rDest
	     <<" => "<<dest.Pt()<<" "<<dest.Phi()<<" "<<dest.z()
	     <<std::endl;
  }
  return dest;

}

std::pair<TVector3,TVector3> helixPropagateApproxR(const TVector3& r3, const TVector3& p3, double rDest, int q, int& status, bool useClosest, bool verbose)
{
  double epsilon = 0.001;
  double p = p3.Mag();
  double kap = (2.99792458e-3*3.8*q/p);
  
  auto lastR3 = r3;
  auto lastT3 = p3.Unit();
  int nIts = 7;

  while (std::abs(lastR3.Perp() - rDest) > epsilon && nIts >= 0){
    auto lineEst = linePropagateR(lastR3, lastT3*p, rDest, status, useClosest, verbose);
    if (status){
      if (verbose) std::cout<<" failed with status "<<status<<std::endl;
      return { lineEst, lastT3*p};
    }
    if (q==0) return {lineEst, lastT3*p};
    
    double dir = (lineEst.x() - lastR3.x())*lastT3.x() + (lineEst.y() - lastR3.y())*lastT3.y() > 0 ? 1. : -1;
    double dS = (lineEst - lastR3).Mag()*dir;
    double phi = kap*dS;
    if (std::abs(phi) > 1) {
      if (verbose) std::cout<<" return line for very large angle "<<status<<std::endl;
      return { lineEst, lastT3*p};
    }
    double alpha = 1 - sin(phi)/phi;
    double beta = (1 - cos(phi))/phi;
    
    TVector3 tau = lastT3; 
    
    TVector3 hEstR3(tau.x()*(1.-alpha) + tau.y()*beta, tau.y()*(1.-alpha) - tau.x()*beta, tau.z());
    hEstR3 *= dS;
    hEstR3 += lastR3;
    lastR3 = hEstR3;
    
    TVector3 hEstT3(tau.x()*cos(phi) + tau.y()*sin(phi), tau.y()*cos(phi) - tau.x()*sin(phi), tau.z());
    lastT3 = hEstT3;
    --nIts;
    if (verbose){
      std::cout<<"nIts "<<nIts<<" rDest "<<rDest<<" dS "<<dS<<" phi "<<phi
	       <<" r3In ("<<r3.Pt()<<", "<<r3.Eta()<<", "<<r3.Phi()<<")"
	       <<" p3In ("<<p3.Pt()<<", "<<p3.Eta()<<", "<<p3.Phi()<<")"
	       <<" r3out ("<<lastR3.Pt()<<", "<<lastR3.Eta()<<", "<<lastR3.Phi()<<")"
	       <<" p3Out ("<<lastT3.Pt()*p<<", "<<lastT3.Eta()<<", "<<lastT3.Phi()<<")"
	       <<std::endl;
    }
  }
  status = (std::abs(lastR3.Perp() - rDest) > epsilon);
  return {lastR3, lastT3*p};
  
}


void fitCircle(std::vector<float> xs, std::vector<float> ys)
{

    TCanvas *c1 = new TCanvas("c1","c1",600,600);
    c1->SetGrid();

    // Generate graph that contains the data
    TGraph* gr = new TGraph(xs.size());
    for (unsigned int i = 0; i < xs.size(); ++i)
    {
        gr->SetPoint(i,xs[i],ys[i]);
    }

    c1->DrawFrame(-120,-120,120,120);
    gr->Draw("p");

    auto chi2Function = [&](const Double_t *par) {
        // minimisation function computing the sum of squares of residuals
        // looping at the graph points
        Int_t np = gr->GetN();
        Double_t f = 0;
        Double_t *x = gr->GetX();
        Double_t *y = gr->GetY();
        for (Int_t i=0;i<np;i++) {
            Double_t u = x[i] - par[0];
            Double_t v = y[i] - par[1];
            Double_t dr = par[2] - std::sqrt(u*u+v*v);
            f += dr*dr;
        }
        return f;
    };

    // wrap chi2 funciton in a function object for the fit
    // 3 is the number of fit parameters (size of array par)
    ROOT::Math::Functor fcn(chi2Function,3);
    ROOT::Fit::Fitter  fitter;
    double pStart[3] = {0,0,1};
    fitter.SetFCN(fcn, pStart);
    fitter.Config().ParSettings(0).SetName("x0");
    fitter.Config().ParSettings(1).SetName("y0");
    fitter.Config().ParSettings(2).SetName("R");

    // do the fit 
    bool ok = fitter.FitFCN();
    if (!ok) {
        Error("line3Dfit","Line3D Fit failed");
    }   

    const ROOT::Fit::FitResult & result = fitter.Result();

    result.Print(std::cout);

    //Draw the circle on top of the points
    TArc *arc = new TArc(result.Parameter(0),result.Parameter(1),result.Parameter(2));
    arc->SetLineColor(kRed);
    arc->SetLineWidth(4);
    arc->SetFillColorAlpha(0, 0.35);
    arc->Draw();
    c1->SaveAs("result.pdf");

}

//__________________________________________________________________________________________
void printMiniDoubletConnectionMultiplicitiesBarrel(SDL::Event& event, int layer, int depth, bool goinside)
{

    std::vector<int> multiplicities;
    int total_nmult = 0;
    float avg_mult = 0;

    if (not goinside)
    {
        // ----------------
        multiplicities.clear();
        total_nmult = 0;
        for (auto& miniDoubletPtr : event.getLayer(layer, SDL::Layer::Barrel).getMiniDoubletPtrs())
        {
            int nmult = 0;
            for (auto& seg1 : miniDoubletPtr->getListOfOutwardSegmentPtrs())
            {
                if (depth == 1)
                    nmult++;
                for (auto& seg2 : seg1->outerMiniDoubletPtr()->getListOfOutwardSegmentPtrs())
                {
                    if (depth == 2)
                        nmult++;
                    for (auto& seg3 : seg2->outerMiniDoubletPtr()->getListOfOutwardSegmentPtrs())
                    {
                        if (depth == 3)
                            nmult++;
                        for (auto& seg4 : seg3->outerMiniDoubletPtr()->getListOfOutwardSegmentPtrs())
                        {
                            if (depth == 4)
                                nmult++;
                            for (auto& seg5 : seg4->outerMiniDoubletPtr()->getListOfOutwardSegmentPtrs())
                            {
                                if (depth == 5)
                                    nmult++;
                            }
                        }
                    }
                }
            }
            multiplicities.push_back(nmult);
            total_nmult += nmult;
        }
        avg_mult = ((float) total_nmult) / ((float) multiplicities.size());
        std::cout <<  " layer: " << layer <<  " depth: " << depth <<  " total_nmult: " << total_nmult <<  " avg_mult: " << avg_mult <<  " goinside: " << goinside <<  std::endl;
    }
    else
    {

        // ----------------
        multiplicities.clear();
        total_nmult = 0;
        for (auto& miniDoubletPtr : event.getLayer(layer, SDL::Layer::Barrel).getMiniDoubletPtrs())
        {
            int nmult = 0;
            for (auto& seg1 : miniDoubletPtr->getListOfInwardSegmentPtrs())
            {
                if (depth == 1)
                    nmult++;
                for (auto& seg2 : seg1->innerMiniDoubletPtr()->getListOfInwardSegmentPtrs())
                {
                    if (depth == 2)
                        nmult++;
                    for (auto& seg3 : seg2->innerMiniDoubletPtr()->getListOfInwardSegmentPtrs())
                    {
                        if (depth == 3)
                            nmult++;
                        for (auto& seg4 : seg3->innerMiniDoubletPtr()->getListOfInwardSegmentPtrs())
                        {
                            if (depth == 4)
                                nmult++;
                            for (auto& seg5 : seg4->innerMiniDoubletPtr()->getListOfInwardSegmentPtrs())
                            {
                                if (depth == 5)
                                    nmult++;
                            }
                        }
                    }
                }
            }
            multiplicities.push_back(nmult);
            total_nmult += nmult;
        }
        avg_mult = ((float) total_nmult) / ((float) multiplicities.size());
        std::cout <<  " layer: " << layer <<  " depth: " << depth <<  " total_nmult: " << total_nmult <<  " avg_mult: " << avg_mult <<  " goinside: " << goinside <<  std::endl;

    }

}

//__________________________________________________________________________________________
vector<int> matchedSimTrkIdxs(SDL::TrackCandidate* tc)
{

    std::vector<int> hitidxs = {
        tc->innerTrackletPtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx(),
        tc->innerTrackletPtr()->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx(),
        tc->innerTrackletPtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx(),
        tc->innerTrackletPtr()->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx(),
        tc->outerTrackletPtr()->innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx(),
        tc->outerTrackletPtr()->innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx(),
        tc->outerTrackletPtr()->innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx(),
        tc->outerTrackletPtr()->innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx(),
        tc->outerTrackletPtr()->outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx(),
        tc->outerTrackletPtr()->outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx(),
        tc->outerTrackletPtr()->outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx(),
        tc->outerTrackletPtr()->outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx()
        };

    std::vector<vector<int>> simtrk_idxs;
    std::vector<int> unique_idxs; // to aggregate which ones to count and test

    for (auto& hitidx : hitidxs)
    {
        std::vector<int> simtrk_idxs_per_hit;
        for (auto& simhit_idx : trk.ph2_simHitIdx()[hitidx])
        {
            int simtrk_idx = trk.simhit_simTrkIdx()[simhit_idx];
            simtrk_idxs_per_hit.push_back(simtrk_idx);
            if (std::find(unique_idxs.begin(), unique_idxs.end(), simtrk_idx) == unique_idxs.end())
                unique_idxs.push_back(simtrk_idx);
        }
        if (simtrk_idxs_per_hit.size() == 0)
        {
            simtrk_idxs_per_hit.push_back(-1);
            if (std::find(unique_idxs.begin(), unique_idxs.end(), -1) == unique_idxs.end())
                unique_idxs.push_back(-1);
        }
        simtrk_idxs.push_back(simtrk_idxs_per_hit);
    }

    // // print
    // std::cout << "va print" << std::endl;
    // for (auto& vec : simtrk_idxs)
    // {
    //     for (auto& idx : vec)
    //     {
    //         std::cout << idx << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "va print end" << std::endl;

    // Compute all permutations
    std::function<void(vector<vector<int>>&, vector<int>, size_t, vector<vector<int>>&)> perm = [&](vector<vector<int>>& result, vector<int> intermediate, size_t n, vector<vector<int>>& va)
    {
        if (va.size() > n)
        {
            for (auto x : va[n])
            {
                intermediate.push_back(x);
                perm(result, intermediate, n+1, va);
            }
        }
        else
        {
            result.push_back(intermediate);
        }
    };

    vector<vector<int>> allperms;
    perm(allperms, vector<int>(), 0, simtrk_idxs);

    // for (auto& iperm : allperms)
    // {
    //     for (auto& idx : iperm)
    //         std::cout << idx << " ";
    //     std::cout << std::endl;
    // }

    std::vector<int> matched_sim_trk_idxs;
    for (auto& trkidx_perm : allperms)
    {
        std::vector<int> counts;
        for (auto& unique_idx : unique_idxs)
        {
            int cnt = std::count(trkidx_perm.begin(), trkidx_perm.end(), unique_idx);
            counts.push_back(cnt);
        }
        auto result = std::max_element(counts.begin(), counts.end());
        int rawidx = std::distance(counts.begin(), result);
        int trkidx = unique_idxs[rawidx];
        if (trkidx < 0)
            continue;
        if (counts[rawidx] > 9)
            matched_sim_trk_idxs.push_back(trkidx);
    }

    return matched_sim_trk_idxs;

}

//__________________________________________________________________________________________
std::vector<float> getPtBounds()
{
    std::vector<float> pt_boundaries;
    if (ana.ptbound_mode == 0)
        pt_boundaries = {0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 10, 15., 25, 50};
    else if (ana.ptbound_mode == 1)
        pt_boundaries = {0.988, 0.99, 0.992, 0.994, 0.996, 0.998, 1.0, 1.002, 1.004, 1.006, 1.008, 1.01, 1.012}; // lowpt
    else if (ana.ptbound_mode == 2)
        pt_boundaries = {0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.00, 1.005, 1.01, 1.015, 1.02, 1.025, 1.03, 1.035, 1.04, 1.045, 1.05}; // pt 0p95 1p05
    else if (ana.ptbound_mode == 3)
        pt_boundaries = {0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.2, 1.5}; // lowpt
    else if (ana.ptbound_mode == 4)
        pt_boundaries = {0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3, 1.32, 1.34, 1.36, 1.38, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0}; // lowpt
    else if (ana.ptbound_mode == 5)
        pt_boundaries = {0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.24, 1.28, 1.32, 1.36, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0}; // lowpt
    else if (ana.ptbound_mode == 6)
        pt_boundaries = {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0, 4.0, 5.0}; // lowpt
    else if (ana.ptbound_mode == 7)
        pt_boundaries = {0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3, 1.32, 1.34, 1.36, 1.38, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50}; // lowpt
    return pt_boundaries;
}

//__________________________________________________________________________________________
bool goodEvent()
{
    if (ana.specific_event_index >= 0)
    {
        if ((int)ana.looper.getCurrentEventIndex() != ana.specific_event_index)
            return false;
    }

    // If splitting jobs are requested then determine whether to process the event or not based on remainder
    if (ana.nsplit_jobs >= 0 and ana.job_index >= 0)
    {
        if (ana.looper.getNEventsProcessed() % ana.nsplit_jobs != (unsigned int) ana.job_index)
            return false;
    }

    if (ana.verbose) std::cout <<  " ana.looper.getCurrentEventIndex(): " << ana.looper.getCurrentEventIndex() <<  std::endl;

    return true;
}

//__________________________________________________________________________________________
bool inTimeTrackWithPdgId(int isimtrk, int pdgid)
{
    // Then select all charged particle
    if (pdgid == 0)
    {
        // Select only muon tracks
        if (abs(trk.sim_q()[isimtrk]) == 0)
            return false;
    }
    else
    {
        // Select tracks with given pdgid
        if (abs(trk.sim_pdgId()[isimtrk]) != pdgid)
            return false;
    }

    // Select in time only
    if (abs(trk.sim_bunchCrossing()[isimtrk]) != 0)
        return false;

    return true;
}

//__________________________________________________________________________________________
TrackletType getTrackletCategory(SDL::Tracklet& tl)
{
    const SDL::Module& innerSgInnerMDAnchorHitModule = tl.innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule();
    const SDL::Module& outerSgInnerMDAnchorHitModule = tl.outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule();
    const SDL::Module& innerSgOuterMDAnchorHitModule = tl.innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule();
    const SDL::Module& outerSgOuterMDAnchorHitModule = tl.outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule();

    const int innerLayerIdx = innerSgInnerMDAnchorHitModule.layer();
    const int outerLayerIdx = outerSgInnerMDAnchorHitModule.layer();

    const bool l1_Barrel = (innerSgInnerMDAnchorHitModule.subdet() == SDL::Module::Barrel);
    const bool l2_Barrel = (innerSgOuterMDAnchorHitModule.subdet() == SDL::Module::Barrel);
    const bool l3_Barrel = (outerSgInnerMDAnchorHitModule.subdet() == SDL::Module::Barrel);
    const bool l4_Barrel = (outerSgOuterMDAnchorHitModule.subdet() == SDL::Module::Barrel);
    const bool l1_Endcap = (innerSgInnerMDAnchorHitModule.subdet() == SDL::Module::Endcap);
    const bool l2_Endcap = (innerSgOuterMDAnchorHitModule.subdet() == SDL::Module::Endcap);
    const bool l3_Endcap = (outerSgInnerMDAnchorHitModule.subdet() == SDL::Module::Endcap);
    const bool l4_Endcap = (outerSgOuterMDAnchorHitModule.subdet() == SDL::Module::Endcap);

    if (innerLayerIdx == 1 and outerLayerIdx == 3 and l1_Barrel and l2_Barrel and l3_Barrel and l4_Barrel) return BB1BB3;
    if (innerLayerIdx == 2 and outerLayerIdx == 4 and l1_Barrel and l2_Barrel and l3_Barrel and l4_Barrel) return BB2BB4;
    if (innerLayerIdx == 3 and outerLayerIdx == 5 and l1_Barrel and l2_Barrel and l3_Barrel and l4_Barrel) return BB3BB5;
    if (innerLayerIdx == 1 and outerLayerIdx == 3 and l1_Barrel and l2_Barrel and l3_Barrel and l4_Endcap) return BB1BE3;
    if (innerLayerIdx == 2 and outerLayerIdx == 4 and l1_Barrel and l2_Barrel and l3_Barrel and l4_Endcap) return BB2BE4;
    if (innerLayerIdx == 3 and outerLayerIdx == 5 and l1_Barrel and l2_Barrel and l3_Barrel and l4_Endcap) return BB3BE5;
    if (innerLayerIdx == 1 and outerLayerIdx == 1 and l1_Barrel and l2_Barrel and l3_Endcap and l4_Endcap) return BB1EE3;
    if (innerLayerIdx == 2 and outerLayerIdx == 1 and l1_Barrel and l2_Barrel and l3_Endcap and l4_Endcap) return BB2EE4;
    if (innerLayerIdx == 3 and outerLayerIdx == 1 and l1_Barrel and l2_Barrel and l3_Endcap and l4_Endcap) return BB3EE5;
    if (innerLayerIdx == 1 and outerLayerIdx == 2 and l1_Barrel and l2_Endcap and l3_Endcap and l4_Endcap) return BE1EE3;
    if (innerLayerIdx == 2 and outerLayerIdx == 2 and l1_Barrel and l2_Endcap and l3_Endcap and l4_Endcap) return BE2EE4;
    if (innerLayerIdx == 3 and outerLayerIdx == 2 and l1_Barrel and l2_Endcap and l3_Endcap and l4_Endcap) return BE3EE5;
    if (innerLayerIdx == 1 and outerLayerIdx == 3 and l1_Endcap and l2_Endcap and l3_Endcap and l4_Endcap) return EE1EE3;
    if (innerLayerIdx == 2 and outerLayerIdx == 4 and l1_Endcap and l2_Endcap and l3_Endcap and l4_Endcap) return EE2EE4;
}

//__________________________________________________________________________________________
int getNPSModules(SDL::Tracklet& tl)
{
    const SDL::Module& innerSgInnerMDAnchorHitModule = tl.innerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule();
    const SDL::Module& outerSgInnerMDAnchorHitModule = tl.outerSegmentPtr()->innerMiniDoubletPtr()->anchorHitPtr()->getModule();
    const SDL::Module& innerSgOuterMDAnchorHitModule = tl.innerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule();
    const SDL::Module& outerSgOuterMDAnchorHitModule = tl.outerSegmentPtr()->outerMiniDoubletPtr()->anchorHitPtr()->getModule();

    int nPS = 0;

    if (innerSgInnerMDAnchorHitModule.moduleType() == SDL::Module::PS)
        nPS++;
    if (innerSgOuterMDAnchorHitModule.moduleType() == SDL::Module::PS)
        nPS++;
    if (outerSgInnerMDAnchorHitModule.moduleType() == SDL::Module::PS)
        nPS++;
    if (outerSgOuterMDAnchorHitModule.moduleType() == SDL::Module::PS)
        nPS++;

    return nPS;
}

std::vector<int> matchedSimTrkIdxs(SDL::Tracklet& tl)
{
    std::vector<int> hitidxs = {
        tl.innerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx(),
        tl.innerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx(),
        tl.innerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx(),
        tl.innerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx(),
        tl.outerSegmentPtr()->innerMiniDoubletPtr()->lowerHitPtr()->idx(),
        tl.outerSegmentPtr()->innerMiniDoubletPtr()->upperHitPtr()->idx(),
        tl.outerSegmentPtr()->outerMiniDoubletPtr()->lowerHitPtr()->idx(),
        tl.outerSegmentPtr()->outerMiniDoubletPtr()->upperHitPtr()->idx()
        };

    std::vector<vector<int>> simtrk_idxs;
    std::vector<int> unique_idxs; // to aggregate which ones to count and test

    for (auto& hitidx : hitidxs)
    {
        std::vector<int> simtrk_idxs_per_hit;
        for (auto& simhit_idx : trk.ph2_simHitIdx()[hitidx])
        {
            int simtrk_idx = trk.simhit_simTrkIdx()[simhit_idx];
            simtrk_idxs_per_hit.push_back(simtrk_idx);
            if (std::find(unique_idxs.begin(), unique_idxs.end(), simtrk_idx) == unique_idxs.end())
                unique_idxs.push_back(simtrk_idx);
        }
        if (simtrk_idxs_per_hit.size() == 0)
        {
            simtrk_idxs_per_hit.push_back(-1);
            if (std::find(unique_idxs.begin(), unique_idxs.end(), -1) == unique_idxs.end())
                unique_idxs.push_back(-1);
        }
        simtrk_idxs.push_back(simtrk_idxs_per_hit);
    }

    // // print
    // std::cout << "va print" << std::endl;
    // for (auto& vec : simtrk_idxs)
    // {
    //     for (auto& idx : vec)
    //     {
    //         std::cout << idx << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "va print end" << std::endl;

    // Compute all permutations
    std::function<void(vector<vector<int>>&, vector<int>, size_t, vector<vector<int>>&)> perm = [&](vector<vector<int>>& result, vector<int> intermediate, size_t n, vector<vector<int>>& va)
    {
        if (va.size() > n)
        {
            for (auto x : va[n])
            {
                intermediate.push_back(x);
                perm(result, intermediate, n+1, va);
            }
        }
        else
        {
            result.push_back(intermediate);
        }
    };

    vector<vector<int>> allperms;
    perm(allperms, vector<int>(), 0, simtrk_idxs);

    // for (auto& iperm : allperms)
    // {
    //     for (auto& idx : iperm)
    //         std::cout << idx << " ";
    //     std::cout << std::endl;
    // }

    std::vector<int> matched_sim_trk_idxs;
    for (auto& trkidx_perm : allperms)
    {
        std::vector<int> counts;
        for (auto& unique_idx : unique_idxs)
        {
            int cnt = std::count(trkidx_perm.begin(), trkidx_perm.end(), unique_idx);
            counts.push_back(cnt);
        }
        auto result = std::max_element(counts.begin(), counts.end());
        int rawidx = std::distance(counts.begin(), result);
        int trkidx = unique_idxs[rawidx];
        if (trkidx < 0)
            continue;
        if (counts[rawidx] >= 8)
            matched_sim_trk_idxs.push_back(trkidx);
    }

    return matched_sim_trk_idxs;

}

//__________________________________________________________________________________________
void loadMaps()
{
    SDL::endcapGeometry.load("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/scripts/endcap_orientation_data_v2.txt"); // centroid values added to the map
    SDL::tiltedGeometry.load("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/scripts/tilted_orientation_data.txt");
    // SDL::moduleConnectionMap.load("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/scripts/module_connection_map_data_10_e0_200_100_pt0p8_2p0_400_pt0p8_2p0_nolossers_dxy35cm_endcaplayer2.txt");
    // SDL::moduleConnectionMap.load("data/module_connection_2020_0429.txt");
    SDL::moduleConnectionMap.load("data/module_connection.txt");
    ana.moduleConnectiongMapLoose.load("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/scripts/module_connection_map_data_10_e0_200_100_pt0p8_2p0_400_pt0p8_2p0_nolossers_dxy35cm_endcaplayer2.txt");
}


//__________________________________________________________________________________________
void addOuterTrackerHits(SDL::Event& event)
{
    // Adding hits to modules
    for (auto&& [ihit, data] : iter::enumerate(iter::zip(trk.ph2_x(), trk.ph2_y(), trk.ph2_z(), trk.ph2_subdet(), trk.ph2_detId())))
    {

        auto&& [x, y, z, subdet, detid] = data;

        if (not (subdet == 5 or subdet == 4))
            continue;

        // Takes two arguments, SDL::Hit, and detId
        // SDL::Event internally will structure whether we already have the module instance or we need to create a new one.
        event.addHitToModule(
                // a hit
                SDL::Hit(x, y, z, ihit),
                // add to module with "detId"
                detid
                );

    }
}

//__________________________________________________________________________________________
void addOuterTrackerHitsFromSimTrack(SDL::Event& event, int isimtrk)
{

    // loop over the simulated hits
    for (unsigned int ith_hit = 0; ith_hit < trk.sim_simHitIdx()[isimtrk].size(); ++ith_hit)
    {

        // Retrieve the sim hit idx
        unsigned int simhitidx = trk.sim_simHitIdx()[isimtrk][ith_hit];

        // To clean up low pt junk in mini-doublet efficiency plots
        if (isMuonCurlingHit(isimtrk, ith_hit))
            break;

        // list of reco hit matched to this sim hit
        for (unsigned int irecohit = 0; irecohit < trk.simhit_hitIdx()[simhitidx].size(); ++irecohit)
        {

            // Get the recohit type
            int recohittype = trk.simhit_hitType()[simhitidx][irecohit];

            // Consider only ph2 hits (i.e. outer tracker hits)
            if (recohittype == 4)
            {

                int ihit = trk.simhit_hitIdx()[simhitidx][irecohit];

                event.addHitToModule(
                        // a hit
                        SDL::Hit(trk.ph2_x()[ihit], trk.ph2_y()[ihit], trk.ph2_z()[ihit], ihit),
                        // add to module with "detId"
                        trk.ph2_detId()[ihit]
                        );

            }

        }

    }
}

//__________________________________________________________________________________________
void printHitSummary(SDL::Event& event)
{
    if (ana.verbose != 0) std::cout << "Summary of hits" << std::endl;
    if (ana.verbose != 0) std::cout << "# of Hits: " << event.getNumberOfHits() << std::endl;
    if (ana.verbose != 0) std::cout << "# of Hits in layer 1: " << event.getNumberOfHitsByLayerBarrel(0) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Hits in layer 2: " << event.getNumberOfHitsByLayerBarrel(1) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Hits in layer 3: " << event.getNumberOfHitsByLayerBarrel(2) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Hits in layer 4: " << event.getNumberOfHitsByLayerBarrel(3) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Hits in layer 5: " << event.getNumberOfHitsByLayerBarrel(4) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Hits in layer 6: " << event.getNumberOfHitsByLayerBarrel(5) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Hits Upper Module in layer 1: " << event.getNumberOfHitsByLayerBarrelUpperModule(0) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Hits Upper Module in layer 2: " << event.getNumberOfHitsByLayerBarrelUpperModule(1) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Hits Upper Module in layer 3: " << event.getNumberOfHitsByLayerBarrelUpperModule(2) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Hits Upper Module in layer 4: " << event.getNumberOfHitsByLayerBarrelUpperModule(3) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Hits Upper Module in layer 5: " << event.getNumberOfHitsByLayerBarrelUpperModule(4) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Hits Upper Module in layer 6: " << event.getNumberOfHitsByLayerBarrelUpperModule(5) << std::endl;
}

//__________________________________________________________________________________________
void printMiniDoubletSummary(SDL::Event& event)
{
    if (ana.verbose != 0) std::cout << "# of Mini-doublets produced: " << event.getNumberOfMiniDoublets() << std::endl;
    if (ana.verbose != 0) std::cout << "# of Mini-doublets produced layer 1: " << event.getNumberOfMiniDoubletsByLayerBarrel(0) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Mini-doublets produced layer 2: " << event.getNumberOfMiniDoubletsByLayerBarrel(1) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Mini-doublets produced layer 3: " << event.getNumberOfMiniDoubletsByLayerBarrel(2) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Mini-doublets produced layer 4: " << event.getNumberOfMiniDoubletsByLayerBarrel(3) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Mini-doublets produced layer 5: " << event.getNumberOfMiniDoubletsByLayerBarrel(4) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Mini-doublets produced layer 6: " << event.getNumberOfMiniDoubletsByLayerBarrel(5) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Mini-doublets considered: " << event.getNumberOfMiniDoubletCandidates() << std::endl;
    if (ana.verbose != 0) std::cout << "# of Mini-doublets considered layer 1: " << event.getNumberOfMiniDoubletCandidatesByLayerBarrel(0) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Mini-doublets considered layer 2: " << event.getNumberOfMiniDoubletCandidatesByLayerBarrel(1) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Mini-doublets considered layer 3: " << event.getNumberOfMiniDoubletCandidatesByLayerBarrel(2) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Mini-doublets considered layer 4: " << event.getNumberOfMiniDoubletCandidatesByLayerBarrel(3) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Mini-doublets considered layer 5: " << event.getNumberOfMiniDoubletCandidatesByLayerBarrel(4) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Mini-doublets considered layer 6: " << event.getNumberOfMiniDoubletCandidatesByLayerBarrel(5) << std::endl;
}

//__________________________________________________________________________________________
void printSegmentSummary(SDL::Event& event)
{
    if (ana.verbose != 0) std::cout << "# of Segments produced: " << event.getNumberOfSegments() << std::endl;
    if (ana.verbose != 0) std::cout << "# of Segments produced layer 1-2: " << event.getNumberOfSegmentsByLayerBarrel(0) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Segments produced layer 2-3: " << event.getNumberOfSegmentsByLayerBarrel(1) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Segments produced layer 3-4: " << event.getNumberOfSegmentsByLayerBarrel(2) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Segments produced layer 4-5: " << event.getNumberOfSegmentsByLayerBarrel(3) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Segments produced layer 5-6: " << event.getNumberOfSegmentsByLayerBarrel(4) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of Segments produced layer 6: " << event.getNumberOfSegmentsByLayerBarrel(5) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Segments considered: " << event.getNumberOfSegmentCandidates() << std::endl;
    if (ana.verbose != 0) std::cout << "# of Segments considered layer 1-2: " << event.getNumberOfSegmentCandidatesByLayerBarrel(0) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Segments considered layer 2-3: " << event.getNumberOfSegmentCandidatesByLayerBarrel(1) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Segments considered layer 3-4: " << event.getNumberOfSegmentCandidatesByLayerBarrel(2) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Segments considered layer 4-5: " << event.getNumberOfSegmentCandidatesByLayerBarrel(3) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Segments considered layer 5-6: " << event.getNumberOfSegmentCandidatesByLayerBarrel(4) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of Segments considered layer 6: " << event.getNumberOfSegmentCandidatesByLayerBarrel(5) << std::endl;
}

//__________________________________________________________________________________________
void printMiniDoubletConnectionMultiplicitiesSummary(SDL::Event& event)
{
    if (ana.verbose != 0) std::cout << "Printing connection information" << std::endl;
    if (ana.verbose != 0)
    {
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 1, 1);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 1, 2);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 1, 3);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 1, 4);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 1, 5);
        std::cout << "--------" << std::endl;

        printMiniDoubletConnectionMultiplicitiesBarrel(event, 2, 1, true);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 2, 1);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 2, 2);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 2, 3);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 2, 4);
        std::cout << "--------" << std::endl;

        printMiniDoubletConnectionMultiplicitiesBarrel(event, 3, 2, true);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 3, 1, true);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 3, 1);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 3, 2);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 3, 3);
        std::cout << "--------" << std::endl;

        printMiniDoubletConnectionMultiplicitiesBarrel(event, 4, 3, true);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 4, 2, true);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 4, 1, true);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 4, 1);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 4, 2);
        std::cout << "--------" << std::endl;

        printMiniDoubletConnectionMultiplicitiesBarrel(event, 5, 4, true);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 5, 3, true);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 5, 2, true);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 5, 1, true);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 5, 1);
        std::cout << "--------" << std::endl;

        printMiniDoubletConnectionMultiplicitiesBarrel(event, 6, 5, true);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 6, 4, true);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 6, 3, true);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 6, 2, true);
        printMiniDoubletConnectionMultiplicitiesBarrel(event, 6, 1, true);
        std::cout << "--------" << std::endl;
    }
}

//__________________________________________________________________________________________
void printTripletSummary(SDL::Event& event)
{
    // ----------------
    if (ana.verbose != 0) std::cout << "# of Triplets produced: " << event.getNumberOfTriplets() << std::endl;
    if (ana.verbose != 0) std::cout << "# of Triplets produced layer 1-2-3: " << event.getNumberOfTripletsByLayerBarrel(0) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Triplets produced layer 2-3-4: " << event.getNumberOfTripletsByLayerBarrel(1) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Triplets produced layer 3-4-5: " << event.getNumberOfTripletsByLayerBarrel(2) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Triplets produced layer 4-5-6: " << event.getNumberOfTripletsByLayerBarrel(3) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of Triplets produced layer 5: " << event.getNumberOfTripletsByLayerBarrel(4) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of Triplets produced layer 6: " << event.getNumberOfTripletsByLayerBarrel(5) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Triplets considered: " << event.getNumberOfTripletCandidates() << std::endl;
    if (ana.verbose != 0) std::cout << "# of Triplets considered layer 1-2-3: " << event.getNumberOfTripletCandidatesByLayerBarrel(0) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Triplets considered layer 2-3-4: " << event.getNumberOfTripletCandidatesByLayerBarrel(1) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Triplets considered layer 3-4-5: " << event.getNumberOfTripletCandidatesByLayerBarrel(2) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Triplets considered layer 4-5-6: " << event.getNumberOfTripletCandidatesByLayerBarrel(3) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of Triplets considered layer 5: " << event.getNumberOfTripletCandidatesByLayerBarrel(4) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of Triplets considered layer 6: " << event.getNumberOfTripletCandidatesByLayerBarrel(5) << std::endl;
    // ----------------
}

//__________________________________________________________________________________________
void printTrackletSummary(SDL::Event& event)
{
    // ----------------
    if (ana.verbose != 0) std::cout << "# of Tracklets produced: " << event.getNumberOfTracklets() << std::endl;
    if (ana.verbose != 0) std::cout << "# of Tracklets produced layer 1-2-3-4: " << event.getNumberOfTrackletsByLayerBarrel(0) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Tracklets produced layer 2-3-4-5: " << event.getNumberOfTrackletsByLayerBarrel(1) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Tracklets produced layer 3-4-5-6: " << event.getNumberOfTrackletsByLayerBarrel(2) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of Tracklets produced layer 4: " << event.getNumberOfTrackletsByLayerBarrel(3) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of Tracklets produced layer 5: " << event.getNumberOfTrackletsByLayerBarrel(4) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of Tracklets produced layer 6: " << event.getNumberOfTrackletsByLayerBarrel(5) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Tracklets considered: " << event.getNumberOfTrackletCandidates() << std::endl;
    if (ana.verbose != 0) std::cout << "# of Tracklets considered layer 1-2-3-4: " << event.getNumberOfTrackletCandidatesByLayerBarrel(0) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Tracklets considered layer 2-3-4-5: " << event.getNumberOfTrackletCandidatesByLayerBarrel(1) << std::endl;
    if (ana.verbose != 0) std::cout << "# of Tracklets considered layer 3-4-5-6: " << event.getNumberOfTrackletCandidatesByLayerBarrel(2) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of Tracklets considered layer 4: " << event.getNumberOfTrackletCandidatesByLayerBarrel(3) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of Tracklets considered layer 5: " << event.getNumberOfTrackletCandidatesByLayerBarrel(4) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of Tracklets considered layer 6: " << event.getNumberOfTrackletCandidatesByLayerBarrel(5) << std::endl;
    // ----------------
}

//__________________________________________________________________________________________
void printTrackCandidateSummary(SDL::Event& event)
{
    // ----------------
    if (ana.verbose != 0) std::cout << "# of TrackCandidates produced: " << event.getNumberOfTrackCandidates() << std::endl;
    if (ana.verbose != 0) std::cout << "# of TrackCandidates produced layer 1-2-3-4-5-6: " << event.getNumberOfTrackCandidatesByLayerBarrel(0) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of TrackCandidates produced layer 2: " << event.getNumberOfTrackCandidatesByLayerBarrel(1) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of TrackCandidates produced layer 3: " << event.getNumberOfTrackCandidatesByLayerBarrel(2) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of TrackCandidates produced layer 4: " << event.getNumberOfTrackCandidatesByLayerBarrel(3) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of TrackCandidates produced layer 5: " << event.getNumberOfTrackCandidatesByLayerBarrel(4) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of TrackCandidates produced layer 6: " << event.getNumberOfTrackCandidatesByLayerBarrel(5) << std::endl;
    if (ana.verbose != 0) std::cout << "# of TrackCandidates considered: " << event.getNumberOfTrackCandidateCandidates() << std::endl;
    if (ana.verbose != 0) std::cout << "# of TrackCandidates considered layer 1-2-3-4-5-6: " << event.getNumberOfTrackCandidateCandidatesByLayerBarrel(0) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of TrackCandidates considered layer 2: " << event.getNumberOfTrackCandidateCandidatesByLayerBarrel(1) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of TrackCandidates considered layer 3: " << event.getNumberOfTrackCandidateCandidatesByLayerBarrel(2) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of TrackCandidates considered layer 4: " << event.getNumberOfTrackCandidateCandidatesByLayerBarrel(3) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of TrackCandidates considered layer 5: " << event.getNumberOfTrackCandidateCandidatesByLayerBarrel(4) << std::endl;
    // if (ana.verbose != 0) std::cout << "# of TrackCandidates considered layer 6: " << event.getNumberOfTrackCandidateCandidatesByLayerBarrel(5) << std::endl;
    // ----------------

}

//__________________________________________________________________________________________
void runMiniDoublet(SDL::Event& event)
{
    TStopwatch my_timer;
    if (ana.verbose != 0) std::cout << "Reco Mini-Doublet start" << std::endl;
    my_timer.Start();
    event.createMiniDoublets();
    float md_elapsed = my_timer.RealTime();
    if (ana.verbose != 0) std::cout << "Reco Mini-doublet processing time: " << md_elapsed << " secs" << std::endl;
}

//__________________________________________________________________________________________
void runSegment(SDL::Event& event)
{
    TStopwatch my_timer;
    if (ana.verbose != 0) std::cout << "Reco Segment start" << std::endl;
    my_timer.Start(kFALSE);
    event.createSegmentsWithModuleMap();
    float sg_elapsed = my_timer.RealTime();
    if (ana.verbose != 0) std::cout << "Reco Segment processing time: " << sg_elapsed << " secs" << std::endl;
}

//__________________________________________________________________________________________
void runTriplet(SDL::Event& event)
{
    TStopwatch my_timer;
    if (ana.verbose != 0) std::cout << "Reco Triplet start" << std::endl;
    my_timer.Start(kFALSE);
    event.createTriplets();
    float tp_elapsed = my_timer.RealTime();
    if (ana.verbose != 0) std::cout << "Reco Triplet processing time: " << tp_elapsed << " secs" << std::endl;
}

//__________________________________________________________________________________________
void runTracklet(SDL::Event& event)
{
    TStopwatch my_timer;
    if (ana.verbose != 0) std::cout << "Reco Tracklet start" << std::endl;
    my_timer.Start(kFALSE);
    // event.createTracklets();
    // event.createTrackletsWithModuleMap();
    event.createTrackletsViaNavigation();
    float tl_elapsed = my_timer.RealTime();
    if (ana.verbose != 0) std::cout << "Reco Tracklet processing time: " << tl_elapsed << " secs" << std::endl;
}

//__________________________________________________________________________________________
void runTrackCandidate(SDL::Event& event)
{
    TStopwatch my_timer;
    if (ana.verbose != 0) std::cout << "Reco TrackCandidate start" << std::endl;
    my_timer.Start(kFALSE);
    // event.createTrackCandidatesFromTriplets();
    // event.createTrackCandidates();
    event.createTrackCandidatesFromTracklets();
    float tc_elapsed = my_timer.RealTime();
    if (ana.verbose != 0) std::cout << "Reco TrackCandidate processing time: " << tc_elapsed << " secs" << std::endl;
}

//__________________________________________________________________________________________
void runSDL(SDL::Event& event)
{

    printHitSummary(event);
    runMiniDoublet(event);
    printMiniDoubletSummary(event);
    runSegment(event);
    printSegmentSummary(event);
    runTriplet(event);
    printTripletSummary(event);
    runTracklet(event);
    printTrackletSummary(event);
    runTrackCandidate(event);
    printTrackCandidateSummary(event);

}

