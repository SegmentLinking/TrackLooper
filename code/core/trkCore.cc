#include "trkCore.h"

//float simhit_p(unsigned int simhitidx)
//{
//
//    // |momentum| calculation
//
//    float px = trk.simhit_px()[simhitidx];
//    float py = trk.simhit_py()[simhitidx];
//    float pz = trk.simhit_pz()[simhitidx];
//    return sqrt(px*px + py*py + pz*pz);
//}
//
////__________________________________________________________________________________________
//float hitAngle(unsigned int simhitidx)
//{
//
//    // This is the angle calculation between position vector and the momentum vector
//
//    float x = trk.simhit_x()[simhitidx];
//    float y = trk.simhit_y()[simhitidx];
//    float z = trk.simhit_z()[simhitidx];
//    float r3 = sqrt(x*x + y*y + z*z);
//    float px = trk.simhit_px()[simhitidx];
//    float py = trk.simhit_py()[simhitidx];
//    float pz = trk.simhit_pz()[simhitidx];
//    float p = sqrt(px*px + py*py + pz*pz);
//    float rdotp = x*px + y*py + z*pz;
//    rdotp = rdotp / r3;
//    rdotp = rdotp / p;
//    float angle = acos(rdotp);
//    return angle;
//}
//
////__________________________________________________________________________________________
//bool isMuonCurlingHit(unsigned int isimtrk, unsigned int ith_hit)
//{
//
//    // To assess whether the ith_hit for isimtrk is a "curling" hit
//
//    // Retrieve the sim hit idx
//    unsigned int simhitidx = trk.sim_simHitIdx()[isimtrk][ith_hit];
//
//    // We're only concerned about hits in the outer tracker
//    // This is more of a sanity check
//    if (not (trk.simhit_subdet()[simhitidx] == 4 or trk.simhit_subdet()[simhitidx] == 5))
//        return false;
//
//    // Retrieve the sim hit pdgId
//    int simhit_particle = trk.simhit_particle()[simhitidx];
//
//    // If the hit is not muon then we can't tell anything
//    if (abs(simhit_particle) != 13)
//        return false;
//
//    // Get the angle
//    float angle = hitAngle(simhitidx);
//
//    // If the angle is quite different then it's the last hit
//    if (abs(angle) > 1.6)
//        return true;
//
//    // Afterwards, we check the energy loss
//    //
//    // If it is the first hit without any previous hit present,
//    // we can't tell whether it is last hit or not
//    // so we just say false to be conservative
//    if (ith_hit == 0)
//        return false;
//
//    // Retrieve the module where the hit lies
//    int detid = trk.simhit_detId()[simhitidx];
//    SDL::Module module = SDL::Module(detid);
//
//    // Calculate the momentum
//    float p = simhit_p(simhitidx);
//
//    // Find the previous simhit that is on the lower side of the module
//    int simhitidx_previous = -999;
//    for (unsigned int ith_back = 1; ith_back <= ith_hit; ith_back++)
//    {
//        // Retrieve the hit idx of ith_hit - ith_back;
//        unsigned int simhitidx_previous_candidate = trk.sim_simHitIdx()[isimtrk][ith_hit-ith_back];
//
//        if (not (trk.simhit_subdet()[simhitidx_previous_candidate] == 4 or trk.simhit_subdet()[simhitidx_previous_candidate] == 5))
//            continue;
//
//        if (not (trk.simhit_particle()[simhitidx_previous_candidate] == 13))
//            continue;
//
//        // Retrieve the module where the previous candidate hit lies
//        int detid = trk.simhit_detId()[simhitidx_previous_candidate];
//        SDL::Module module = SDL::Module(detid);
//
//        // If the module is lower, then we get the index
//        if (module.isLower())
//        {
//            simhitidx_previous = simhitidx_previous_candidate;
//            break;
//        }
//
//    }
//
//    // If no previous layer is found then can't do much
//    if (simhitidx_previous == -999)
//        return false;
//
//    // Get the previous layer momentum
//    float p_previous = simhit_p(simhitidx_previous);
//
//    // Calculate the momentum loss
//    float loss = fabs(p_previous - p) / p_previous;
//
//    // If the momentum loss is large it is the last hit
//    if (loss > 0.35)
//        return true;
//
//    // If it reaches this point again, we're not sure what this hit is
//    // So we return false
//    return false;
//
//}




bool hasAll12HitsWithNBarrelUsingModuleMap(unsigned int isimtrk, int nbarrel, bool usesimhits)
{

    // Select only tracks that left all 12 hits in the barrel
    std::array<std::vector<SDL::CPU::Module>, 6> layers_modules_barrel; // Watch out for duplicates in this vector, do not count with this for unique count.
    std::array<std::vector<SDL::CPU::Module>, 6> layers_modules_endcap; // Watch out for duplicates in this vector, do not count with this for unique count.

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

        //if (isMuonCurlingHit(isimtrk, ith_hit))
        //    break;

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
                        layers_modules_barrel[trk.ph2_layer()[ihit] - 1].push_back(SDL::CPU::Module(trk.ph2_detId()[ihit]));
                    }
                    if (trk.ph2_subdet()[ihit] == 4)
                    {
                        layers_modules_endcap[trk.ph2_layer()[ihit] - 1].push_back(SDL::CPU::Module(trk.ph2_detId()[ihit]));
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
                    layers_modules_barrel[trk.simhit_layer()[simhitidx] - 1].push_back(SDL::CPU::Module(trk.simhit_detId()[simhitidx]));
                }
                if (trk.simhit_subdet()[simhitidx] == 4)
                {
                    layers_modules_endcap[trk.simhit_layer()[simhitidx] - 1].push_back(SDL::CPU::Module(trk.simhit_detId()[simhitidx]));
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
        const std::array<std::vector<SDL::CPU::Module>, 6>& layers_modules = is_ilayer_barrel ? layers_modules_barrel : layers_modules_endcap;

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



bool hasAll12HitsWithNBarrel(unsigned int isimtrk, int nbarrel)
{

    // Select only tracks that left all 12 hits in the barrel
    std::array<std::vector<SDL::CPU::Module>, 6> layers_modules_barrel;
    std::array<std::vector<SDL::CPU::Module>, 6> layers_modules_endcap;

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

        //if (isMuonCurlingHit(isimtrk, ith_hit))
        //    break;

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
                    layers_modules_barrel[trk.ph2_layer()[ihit] - 1].push_back(SDL::CPU::Module(trk.ph2_detId()[ihit]));
                }
                if (trk.ph2_subdet()[ihit] == 4)
                {
                    layers_modules_endcap[trk.ph2_layer()[ihit] - 1].push_back(SDL::CPU::Module(trk.ph2_detId()[ihit]));
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


    //float pt = trk.sim_pt()[isimtrk];
    //float eta = trk.sim_eta()[isimtrk];

    // if (abs((trk.sim_pt()[isimtrk] - 0.71710)) < 0.00001)
    // {
    //     std::cout << std::endl;
    //     std::cout <<  " has_good_pair_by_layer[0]: " << has_good_pair_by_layer[0] <<  " has_good_pair_by_layer[1]: " << has_good_pair_by_layer[1] <<  " has_good_pair_by_layer[2]: " << has_good_pair_by_layer[2] <<  " has_good_pair_by_layer[3]: " << has_good_pair_by_layer[3] <<  " has_good_pair_by_layer[4]: " << has_good_pair_by_layer[4] <<  " has_good_pair_by_layer[5]: " << has_good_pair_by_layer[5] <<  " pt: " << pt <<  " eta: " << eta <<  std::endl;
    // }

    return has_good_pair_all_layer;

}


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
//float addOuterTrackerHits(SDL::CPU::Event& event)
//{
//
//    TStopwatch my_timer;
//    if (ana.verbose >= 2) std::cout << "Loading Outer Tracker Hits for CPU...." << std::endl;
//    my_timer.Start();
//
//    // Adding hits to modules
//    for (auto&& [ihit, data] : iter::enumerate(iter::zip(trk.ph2_x(), trk.ph2_y(), trk.ph2_z(), trk.ph2_subdet(), trk.ph2_detId())))
//    {
//
//        auto&& [x, y, z, subdet, detid] = data;
//
//        if (not (subdet == 5 or subdet == 4))
//            continue;
//
//        // Takes two arguments, SDL::Hit, and detId
//        // SDL::Event internally will structure whether we already have the module instance or we need to create a new one.
//        event.addHitToModule(
//                // a hit
//                SDL::CPU::Hit(x, y, z, ihit),
//                // add to module with "detId"
//                detid
//                );
//
//    }
//
//    float hit_loading_elapsed = my_timer.RealTime();
//    if (ana.verbose >= 2) std::cout << "Loading outer tracker hits processing time: " << hit_loading_elapsed << " secs" << std::endl;
//    return hit_loading_elapsed;
//}
//
////__________________________________________________________________________________________
//float addOuterTrackerSimHits(SDL::CPU::Event& event)
//{
//
//    TStopwatch my_timer;
//    if (ana.verbose >= 2) std::cout << "Loading Outer Tracker Hits for CPU...." << std::endl;
//    my_timer.Start();
//
//    // Adding hits to modules
//    for (auto&& [ihit, data] : iter::enumerate(iter::zip(trk.simhit_x(), trk.simhit_y(), trk.simhit_z(), trk.simhit_subdet(), trk.simhit_detId())))
//    {
//
//        auto&& [x, y, z, subdet, detid] = data;
//
//        if (not (subdet == 5 or subdet == 4))
//            continue;
//
//        // Takes two arguments, SDL::Hit, and detId
//        // SDL::Event internally will structure whether we already have the module instance or we need to create a new one.
//        event.addHitToModule(
//                // a hit
//                SDL::CPU::Hit(x, y, z, ihit),
//                // add to module with "detId"
//                detid
//                );
//
//    }
//
//    float hit_loading_elapsed = my_timer.RealTime();
//    if (ana.verbose >= 2) std::cout << "Loading outer tracker hits processing time: " << hit_loading_elapsed << " secs" << std::endl;
//    return hit_loading_elapsed;
//}
//
////__________________________________________________________________________________________
//float addOuterTrackerSimHitsFromPVOnly(SDL::CPU::Event& event)
//{
//
//    TStopwatch my_timer;
//    if (ana.verbose >= 2) std::cout << "Loading Outer Tracker Hits for CPU...." << std::endl;
//    my_timer.Start();
//
//    // Adding hits to modules
//    for (auto&& [ihit, data] : iter::enumerate(iter::zip(trk.simhit_x(), trk.simhit_y(), trk.simhit_z(), trk.simhit_subdet(), trk.simhit_detId())))
//    {
//
//        if (trk.sim_bunchCrossing()[trk.simhit_simTrkIdx()[ihit]] != 0)
//            continue;
//        if (trk.sim_event()[trk.simhit_simTrkIdx()[ihit]] != 0)
//            continue;
//
//        auto&& [x, y, z, subdet, detid] = data;
//
//        if (not (subdet == 5 or subdet == 4))
//            continue;
//
//        // Takes two arguments, SDL::Hit, and detId
//        // SDL::Event internally will structure whether we already have the module instance or we need to create a new one.
//        event.addHitToModule(
//                // a hit
//                SDL::CPU::Hit(x, y, z, ihit),
//                // add to module with "detId"
//                detid
//                );
//
//    }
//
//    float hit_loading_elapsed = my_timer.RealTime();
//    if (ana.verbose >= 2) std::cout << "Loading outer tracker hits processing time: " << hit_loading_elapsed << " secs" << std::endl;
//    return hit_loading_elapsed;
//}
//
////__________________________________________________________________________________________
//float addOuterTrackerSimHitsNotFromPVOnly(SDL::CPU::Event& event)
//{
//
//    TStopwatch my_timer;
//    if (ana.verbose >= 2) std::cout << "Loading Outer Tracker Hits for CPU...." << std::endl;
//    my_timer.Start();
//
//    // Adding hits to modules
//    for (auto&& [ihit, data] : iter::enumerate(iter::zip(trk.simhit_x(), trk.simhit_y(), trk.simhit_z(), trk.simhit_subdet(), trk.simhit_detId())))
//    {
//
//        if (trk.sim_bunchCrossing()[trk.simhit_simTrkIdx()[ihit]] == 0 and trk.sim_event()[trk.simhit_simTrkIdx()[ihit]] == 0)
//            continue;
//
//        auto&& [x, y, z, subdet, detid] = data;
//
//        if (not (subdet == 5 or subdet == 4))
//            continue;
//
//        // Takes two arguments, SDL::Hit, and detId
//        // SDL::Event internally will structure whether we already have the module instance or we need to create a new one.
//        event.addHitToModule(
//                // a hit
//                SDL::CPU::Hit(x, y, z, ihit),
//                // add to module with "detId"
//                detid
//                );
//
//    }
//
//    float hit_loading_elapsed = my_timer.RealTime();
//    if (ana.verbose >= 2) std::cout << "Loading outer tracker hits processing time: " << hit_loading_elapsed << " secs" << std::endl;
//    return hit_loading_elapsed;
//}

//float runMiniDoublet(SDL::Event& event, int evt)
float runMiniDoublet(SDL::Event* event, int evt)
{
    TStopwatch my_timer;
    if (ana.verbose >= 2) std::cout << "Reco Mini-Doublet start " << evt<< std::endl;
    my_timer.Start();
    //event.createMiniDoublets();
    event->createMiniDoublets();
    float md_elapsed = my_timer.RealTime();

    if (ana.verbose >= 2) std::cout << evt<< " Reco Mini-doublet processing time: " << md_elapsed << " secs" << std::endl;

    //if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced: " << event.getNumberOfMiniDoublets() << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced barrel layer 1: " << event.getNumberOfMiniDoubletsByLayerBarrel(0) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced barrel layer 2: " << event.getNumberOfMiniDoubletsByLayerBarrel(1) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced barrel layer 3: " << event.getNumberOfMiniDoubletsByLayerBarrel(2) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced barrel layer 4: " << event.getNumberOfMiniDoubletsByLayerBarrel(3) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced barrel layer 5: " << event.getNumberOfMiniDoubletsByLayerBarrel(4) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced barrel layer 6: " << event.getNumberOfMiniDoubletsByLayerBarrel(5) << std::endl;

    //if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced endcap layer 1: " << event.getNumberOfMiniDoubletsByLayerEndcap(0) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced endcap layer 2: " << event.getNumberOfMiniDoubletsByLayerEndcap(1) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced endcap layer 3: " << event.getNumberOfMiniDoubletsByLayerEndcap(2) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced endcap layer 4: " << event.getNumberOfMiniDoubletsByLayerEndcap(3) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced endcap layer 5: " << event.getNumberOfMiniDoubletsByLayerEndcap(4) << std::endl;

    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced: " << event->getNumberOfMiniDoublets() << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced barrel layer 1: " << event->getNumberOfMiniDoubletsByLayerBarrel(0) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced barrel layer 2: " << event->getNumberOfMiniDoubletsByLayerBarrel(1) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced barrel layer 3: " << event->getNumberOfMiniDoubletsByLayerBarrel(2) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced barrel layer 4: " << event->getNumberOfMiniDoubletsByLayerBarrel(3) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced barrel layer 5: " << event->getNumberOfMiniDoubletsByLayerBarrel(4) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced barrel layer 6: " << event->getNumberOfMiniDoubletsByLayerBarrel(5) << std::endl;

    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced endcap layer 1: " << event->getNumberOfMiniDoubletsByLayerEndcap(0) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced endcap layer 2: " << event->getNumberOfMiniDoubletsByLayerEndcap(1) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced endcap layer 3: " << event->getNumberOfMiniDoubletsByLayerEndcap(2) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced endcap layer 4: " << event->getNumberOfMiniDoubletsByLayerEndcap(3) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced endcap layer 5: " << event->getNumberOfMiniDoubletsByLayerEndcap(4) << std::endl;


    return md_elapsed;

}

//float runSegment(SDL::Event& event)
float runSegment(SDL::Event* event)
{
    TStopwatch my_timer;
    if (ana.verbose >= 2) std::cout << "Reco Segment start" << std::endl;
    my_timer.Start();
    event->createSegmentsWithModuleMap();
    //event.createSegmentsWithModuleMap();
    float sg_elapsed = my_timer.RealTime();
    if (ana.verbose >= 2) std::cout << "Reco Segment processing time: " << sg_elapsed << " secs" << std::endl;

    //if (ana.verbose >= 2) std::cout << "# of Segments produced: " << event.getNumberOfSegments() << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Segments produced layer 1-2: " << event.getNumberOfSegmentsByLayerBarrel(0) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Segments produced layer 2-3: " << event.getNumberOfSegmentsByLayerBarrel(1) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Segments produced layer 3-4: " << event.getNumberOfSegmentsByLayerBarrel(2) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Segments produced layer 4-5: " << event.getNumberOfSegmentsByLayerBarrel(3) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Segments produced layer 5-6: " << event.getNumberOfSegmentsByLayerBarrel(4) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Segments produced endcap layer 1: " << event.getNumberOfSegmentsByLayerEndcap(0) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Segments produced endcap layer 2: " << event.getNumberOfSegmentsByLayerEndcap(1) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Segments produced endcap layer 3: " << event.getNumberOfSegmentsByLayerEndcap(2) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Segments produced endcap layer 4: " << event.getNumberOfSegmentsByLayerEndcap(3) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Segments produced endcap layer 5: " << event.getNumberOfSegmentsByLayerEndcap(4) << std::endl;

    if (ana.verbose >= 2) std::cout << "# of Segments produced: " << event->getNumberOfSegments() << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Segments produced layer 1-2: " << event->getNumberOfSegmentsByLayerBarrel(0) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Segments produced layer 2-3: " << event->getNumberOfSegmentsByLayerBarrel(1) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Segments produced layer 3-4: " << event->getNumberOfSegmentsByLayerBarrel(2) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Segments produced layer 4-5: " << event->getNumberOfSegmentsByLayerBarrel(3) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Segments produced layer 5-6: " << event->getNumberOfSegmentsByLayerBarrel(4) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Segments produced endcap layer 1: " << event->getNumberOfSegmentsByLayerEndcap(0) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Segments produced endcap layer 2: " << event->getNumberOfSegmentsByLayerEndcap(1) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Segments produced endcap layer 3: " << event->getNumberOfSegmentsByLayerEndcap(2) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Segments produced endcap layer 4: " << event->getNumberOfSegmentsByLayerEndcap(3) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Segments produced endcap layer 5: " << event->getNumberOfSegmentsByLayerEndcap(4) << std::endl;


    return sg_elapsed;

}


//float runT3(SDL::Event& event)
float runT3(SDL::Event* event)
{
    TStopwatch my_timer;
    if (ana.verbose >= 2) std::cout << "Reco T3 start" << std::endl;
    my_timer.Start();
    event->createTriplets();
    //event.createTriplets();
    float t3_elapsed = my_timer.RealTime();
    if (ana.verbose >= 2) std::cout << "Reco T3 processing time: " << t3_elapsed<< " secs" << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of T3s produced: " << event.getNumberOfTriplets() << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of T3s produced layer 1-2-3: " << event.getNumberOfTripletsByLayerBarrel(0) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of T3s produced layer 2-3-4: " << event.getNumberOfTripletsByLayerBarrel(1) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of T3s produced layer 3-4-5: " << event.getNumberOfTripletsByLayerBarrel(2) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of T3s produced layer 4-5-6: " << event.getNumberOfTripletsByLayerBarrel(3) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of T3s produced endcap layer 1-2-3: " << event.getNumberOfTripletsByLayerEndcap(0) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of T3s produced endcap layer 2-3-4: " << event.getNumberOfTripletsByLayerEndcap(1) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of T3s produced endcap layer 3-4-5: " << event.getNumberOfTripletsByLayerEndcap(2) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of T3s produced endcap layer 1: " << event.getNumberOfTripletsByLayerEndcap(0) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of T3s produced endcap layer 2: " << event.getNumberOfTripletsByLayerEndcap(1) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of T3s produced endcap layer 3: " << event.getNumberOfTripletsByLayerEndcap(2) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of T3s produced endcap layer 4: " << event.getNumberOfTripletsByLayerEndcap(3) << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of T3s produced endcap layer 5: " << event.getNumberOfTripletsByLayerEndcap(4) << std::endl;

    if (ana.verbose >= 2) std::cout << "# of T3s produced: " << event->getNumberOfTriplets() << std::endl;
    if (ana.verbose >= 2) std::cout << "# of T3s produced layer 1-2-3: " << event->getNumberOfTripletsByLayerBarrel(0) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of T3s produced layer 2-3-4: " << event->getNumberOfTripletsByLayerBarrel(1) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of T3s produced layer 3-4-5: " << event->getNumberOfTripletsByLayerBarrel(2) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of T3s produced layer 4-5-6: " << event->getNumberOfTripletsByLayerBarrel(3) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of T3s produced endcap layer 1-2-3: " << event->getNumberOfTripletsByLayerEndcap(0) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of T3s produced endcap layer 2-3-4: " << event->getNumberOfTripletsByLayerEndcap(1) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of T3s produced endcap layer 3-4-5: " << event->getNumberOfTripletsByLayerEndcap(2) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of T3s produced endcap layer 1: " << event->getNumberOfTripletsByLayerEndcap(0) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of T3s produced endcap layer 2: " << event->getNumberOfTripletsByLayerEndcap(1) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of T3s produced endcap layer 3: " << event->getNumberOfTripletsByLayerEndcap(2) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of T3s produced endcap layer 4: " << event->getNumberOfTripletsByLayerEndcap(3) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of T3s produced endcap layer 5: " << event->getNumberOfTripletsByLayerEndcap(4) << std::endl;

    return t3_elapsed;

}


//float runpT3(SDL::Event& event)
float runpT3(SDL::Event* event)
{
    TStopwatch my_timer;
    if (ana.verbose >= 2) std::cout << "Reco Pixel Triplet pT3 start" << std::endl;
    my_timer.Start();
    event->createPixelTriplets();
    //event.createPixelTriplets();
    float pt3_elapsed = my_timer.RealTime();
    if (ana.verbose >= 2) std::cout << "Reco pT3 processing time: " << pt3_elapsed << " secs" << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Pixel T3s produced: "<< event->getNumberOfPixelTriplets() << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Pixel T3s produced: "<< event.getNumberOfPixelTriplets() << std::endl;

    return pt3_elapsed;
}

float runTrackCandidate(SDL::Event* event)
//float runTrackCandidate(SDL::Event& event)
{
    return runTrackCandidateTest_v2(event);
}

float runQuintuplet(SDL::Event* event)
//float runQuintuplet(SDL::Event& event)
{
     TStopwatch my_timer;
    if (ana.verbose >= 2) std::cout << "Reco Quintuplet start" << std::endl;
    my_timer.Start();
    //event.createQuintuplets();
    event->createQuintuplets();
    float t5_elapsed = my_timer.RealTime();
    if (ana.verbose >= 2) std::cout << "Reco Quintuplet processing time: " << t5_elapsed << " secs" << std::endl;

//    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced: " << event.getNumberOfQuintuplets() << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced layer 1-2-3-4-5-6: " << event.getNumberOfQuintupletsByLayerBarrel(0) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced layer 2: " << event.getNumberOfQuintupletsByLayerBarrel(1) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced layer 3: " << event.getNumberOfQuintupletsByLayerBarrel(2) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced layer 4: " << event.getNumberOfQuintupletsByLayerBarrel(3) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced layer 5: " << event.getNumberOfQuintupletsByLayerBarrel(4) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced layer 6: " << event.getNumberOfQuintupletsByLayerBarrel(5) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced endcap layer 1: " << event.getNumberOfQuintupletsByLayerEndcap(0) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced endcap layer 2: " << event.getNumberOfQuintupletsByLayerEndcap(1) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced endcap layer 3: " << event.getNumberOfQuintupletsByLayerEndcap(2) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced endcap layer 4: " << event.getNumberOfQuintupletsByLayerEndcap(3) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced endcap layer 5: " << event.getNumberOfQuintupletsByLayerEndcap(4) << std::endl;

    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced: " << event->getNumberOfQuintuplets() << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced layer 1-2-3-4-5-6: " << event->getNumberOfQuintupletsByLayerBarrel(0) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced layer 2: " << event->getNumberOfQuintupletsByLayerBarrel(1) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced layer 3: " << event->getNumberOfQuintupletsByLayerBarrel(2) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced layer 4: " << event->getNumberOfQuintupletsByLayerBarrel(3) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced layer 5: " << event->getNumberOfQuintupletsByLayerBarrel(4) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced layer 6: " << event->getNumberOfQuintupletsByLayerBarrel(5) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced endcap layer 1: " << event->getNumberOfQuintupletsByLayerEndcap(0) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced endcap layer 2: " << event->getNumberOfQuintupletsByLayerEndcap(1) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced endcap layer 3: " << event->getNumberOfQuintupletsByLayerEndcap(2) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced endcap layer 4: " << event->getNumberOfQuintupletsByLayerEndcap(3) << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Quintuplets produced endcap layer 5: " << event->getNumberOfQuintupletsByLayerEndcap(4) << std::endl;

    return t5_elapsed;
   
}

//float runPixelLineSegment(SDL::Event& event)
float runPixelLineSegment(SDL::Event* event)
{
    TStopwatch my_timer;
    if (ana.verbose >= 2) std::cout << "Reco Pixel Line Segment start" << std::endl;
    my_timer.Start();
    event->pixelLineSegmentCleaning();
    //event.pixelLineSegmentCleaning();
    float pls_elapsed = my_timer.RealTime();
    if (ana.verbose >= 2) std::cout << "Reco Pixel Line Segment processing time: " << pls_elapsed << " secs" << std::endl;

    return pls_elapsed;
}
float runPixelQuintuplet(SDL::Event* event)
//float runPixelQuintuplet(SDL::Event& event)
{
    TStopwatch my_timer;
    if (ana.verbose >= 2) std::cout << "Reco Pixel Quintuplet start" << std::endl;
    my_timer.Start();
    event->createPixelQuintuplets();
    //event.createPixelQuintuplets();
    float pt5_elapsed = my_timer.RealTime();
    if (ana.verbose >= 2) std::cout << "Reco Pixel Quintuplet processing time: " << pt5_elapsed << " secs" << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Pixel Quintuplets produced: " << event->getNumberOfPixelQuintuplets() << std::endl;
    //if (ana.verbose >= 2) std::cout << "# of Pixel Quintuplets produced: " << event.getNumberOfPixelQuintuplets() << std::endl;

    return pt5_elapsed;
}



float runTrackCandidateTest_v2(SDL::Event* event)
//float runTrackCandidateTest_v2(SDL::Event& event)
{
    TStopwatch my_timer;
    if (ana.verbose >= 2) std::cout << "Reco TrackCandidate start" << std::endl;
    my_timer.Start();
    event->createTrackCandidates();
    //event.createTrackCandidates();
    float tc_elapsed = my_timer.RealTime();
    if (ana.verbose >= 2) std::cout << "Reco TrackCandidate processing time: " << tc_elapsed << " secs" << std::endl;

    if (ana.verbose >= 2) std::cout << "# of TrackCandidates produced: " << event->getNumberOfTrackCandidates() << std::endl;
    if (ana.verbose >= 2) std::cout << "# of Pixel TrackCandidates produced: "<< event->getNumberOfPixelTrackCandidates() << std::endl;
    if (ana.verbose >= 2) std::cout << "    # of pT5 TrackCandidates produced: "<< event->getNumberOfPT5TrackCandidates() << std::endl;
    if (ana.verbose >= 2) std::cout << "    # of pT3 TrackCandidates produced: "<< event->getNumberOfPT3TrackCandidates() << std::endl;
    if (ana.verbose >= 2) std::cout << "    # of pLS TrackCandidates produced: "<< event->getNumberOfPLSTrackCandidates() << std::endl;
    if (ana.verbose >= 2) std::cout << "# of T5 TrackCandidates produced: "<< event->getNumberOfT5TrackCandidates() << std::endl;

    return tc_elapsed;

}

//#ifdef TRACK_EXTENSIONS
float runTrackExtensions(SDL::Event* event)
{
    TStopwatch my_timer;
    if (ana.verbose >= 2) 
    {
        std::cout << "Reco Track Extension start" << std::endl;
    }
    my_timer.Start();
    event->createExtendedTracks();
    float tce_elapsed = my_timer.RealTime();
    if (ana.verbose >= 2)
    {
        std::cout<<"Reco Track Extension processing time: " << tce_elapsed<<" secs "<< std::endl;
        std::cout<<"# of Track Extensions produced: "<<event->getNumberOfExtendedTracks()<<std::endl;

#ifdef T3T3_EXTENSIONS
        std::cout<<"# of T3T3 Track Extensions produced: "<<event->getNumberOfT3T3ExtendedTracks()<<std::endl;
#endif

    }
    return tce_elapsed;
}
//#endif

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

    if (ana.verbose >= 2) std::cout <<  " ana.looper.getCurrentEventIndex(): " << ana.looper.getCurrentEventIndex() <<  std::endl;

    return true;
}

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
    else if (ana.ptbound_mode == 8)
        pt_boundaries = {0, 0.5, 1.0, 3.0, 5.0, 10, 15., 25, 50};
    return pt_boundaries;
}

bool inTimeTrackWithPdgId(int isimtrk, int pdgid)
{
    // Then select all charged particle
    if (pdgid == 0)
    {
        // Select all charged particle tracks
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

std::vector<int> matchedSimTrkIdxs(std::vector<int> hitidxs, std::vector<int> hittypes, bool verbose)
{
    if (hitidxs.size() != hittypes.size())
    {
        std::cout << "Error: matched_sim_trk_idxs()   hitidxs and hittypes have different lengths" << std::endl;
        std::cout << "hitidxs.size(): " << hitidxs.size() << std::endl;
        std::cout << "hittypes.size(): " << hittypes.size() << std::endl;
    }

    std::vector<std::pair<int, int>> to_check_duplicate;
    for (auto&& [ihit, ihitdata] : iter::enumerate(iter::zip(hitidxs, hittypes)))
    {
        auto&& [hitidx, hittype] = ihitdata;
        auto item = std::make_pair(hitidx, hittype);
        if (std::find(to_check_duplicate.begin(), to_check_duplicate.end(), item) == to_check_duplicate.end())
        {
            to_check_duplicate.push_back(item);
        }
    }

    int nhits_input = to_check_duplicate.size();

    std::vector<vector<int>> simtrk_idxs;
    std::vector<int> unique_idxs; // to aggregate which ones to count and test

    if (verbose)
    {
        std::cout <<  " '------------------------': " << "------------------------" <<  std::endl;
    }

    for (auto&& [ihit, ihitdata] : iter::enumerate(to_check_duplicate))
    {
        auto&& [hitidx, hittype] = ihitdata;

        if (verbose)
        {
            std::cout <<  " hitidx: " << hitidx <<  " hittype: " << hittype <<  std::endl;
        }

        std::vector<int> simtrk_idxs_per_hit;

        const std::vector<vector<int>>* simHitIdxs;

        if (hittype == 4)
            simHitIdxs = &trk.ph2_simHitIdx();
        else
            simHitIdxs = &trk.pix_simHitIdx();

        if ( static_cast<const int>((*simHitIdxs).size()) <= hitidx)
        {
            std::cout << "ERROR" << std::endl;
            std::cout <<  " hittype: " << hittype <<  std::endl;
            std::cout <<  " trk.pix_simHitIdx().size(): " << trk.pix_simHitIdx().size() <<  std::endl;
            std::cout <<  " trk.ph2_simHitIdx().size(): " << trk.ph2_simHitIdx().size() <<  std::endl;
            std::cout << (*simHitIdxs).size() << " " << hittype << std::endl;
            std::cout << hitidx << " " << hittype << std::endl;
        }

        for (auto& simhit_idx : (*simHitIdxs).at(hitidx))
        {
            if (static_cast<const int>(trk.simhit_simTrkIdx().size()) <= simhit_idx)
            {
                std::cout << (*simHitIdxs).size() << " " << hittype << std::endl;
                std::cout << hitidx << " " << hittype << std::endl;
                std::cout << trk.simhit_simTrkIdx().size() << " " << simhit_idx << std::endl;
            }
            int simtrk_idx = trk.simhit_simTrkIdx().at(simhit_idx);
            if (verbose)
            {
                std::cout <<  " hitidx: " << hitidx <<  " simhit_idx: " << simhit_idx <<  " simtrk_idx: " << simtrk_idx <<  std::endl;
            }
            simtrk_idxs_per_hit.push_back(simtrk_idx);
            if (std::find(unique_idxs.begin(), unique_idxs.end(), simtrk_idx) == unique_idxs.end())
                unique_idxs.push_back(simtrk_idx);
        }

        if (simtrk_idxs_per_hit.size() == 0)
        {
            if (verbose)
            {
                std::cout <<  " hitidx: " << hitidx <<  " -1: " << -1 <<  std::endl;
            }
            simtrk_idxs_per_hit.push_back(-1);
            if (std::find(unique_idxs.begin(), unique_idxs.end(), -1) == unique_idxs.end())
                unique_idxs.push_back(-1);
        }

        simtrk_idxs.push_back(simtrk_idxs_per_hit);
    }

    if (verbose)
    {
        std::cout <<  " unique_idxs.size(): " << unique_idxs.size() <<  std::endl;
        for (auto& unique_idx : unique_idxs)
        {
            std::cout <<  " unique_idx: " << unique_idx <<  std::endl;
        }
    }

    // print
    if (verbose)
    {
        std::cout << "va print" << std::endl;
        for (auto& vec : simtrk_idxs)
        {
            for (auto& idx : vec)
            {
                std::cout << idx << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "va print end" << std::endl;
    }

    // Compute all permutations
    std::function<void(vector<vector<int>>&, vector<int>, size_t, vector<vector<int>>&)> perm =
        [&](vector<vector<int>>& result, vector<int> intermediate, size_t n, vector<vector<int>>& va)
    {
        // std::cout <<  " 'called': " << "called" <<  std::endl;
        if (va.size() > n)
        {
            for (auto x : va[n])
            {
                // std::cout <<  " n: " << n <<  std::endl;
                // std::cout <<  " intermediate.size(): " << intermediate.size() <<  std::endl;
                std::vector<int> copy_intermediate(intermediate);
                copy_intermediate.push_back(x);
                perm(result, copy_intermediate, n+1, va);
            }
        }
        else
        {
            result.push_back(intermediate);
        }
    };

    vector<vector<int>> allperms;
    perm(allperms, vector<int>(), 0, simtrk_idxs);

    if (verbose)
    {
        std::cout <<  " allperms.size(): " << allperms.size() <<  std::endl;
        for (unsigned iperm = 0; iperm < allperms.size(); ++iperm)
        {
            std::cout <<  " allperms[iperm].size(): " << allperms[iperm].size() <<  std::endl;
            for (unsigned ielem = 0; ielem < allperms[iperm].size(); ++ielem)
            {
                std::cout <<  " allperms[iperm][ielem]: " << allperms[iperm][ielem] <<  std::endl;
            }
        }
    }
    int maxHitMatchCount = 0; //ultimate maximum of the number of matched hits
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
        if (counts[rawidx] > (((float)nhits_input) * 0.75))
            matched_sim_trk_idxs.push_back(trkidx);
        maxHitMatchCount = std::max(maxHitMatchCount, *std::max_element(counts.begin(), counts.end()));
    }
    set<int> s;
    unsigned size = matched_sim_trk_idxs.size();
    for( unsigned i = 0; i < size; ++i ) s.insert( matched_sim_trk_idxs[i] );
    matched_sim_trk_idxs.assign( s.begin(), s.end() );
    return matched_sim_trk_idxs;
}

bool isMTVMatch(unsigned int isimtrk, std::vector<unsigned int> hit_idxs, bool verbose)
{
    std::vector<unsigned int> sim_trk_ihits;
    for (auto& i_simhit_idx : trk.sim_simHitIdx()[isimtrk])
    {
        for (auto& ihit : trk.simhit_hitIdx()[i_simhit_idx])
        {
            sim_trk_ihits.push_back(ihit);
        }
    }

    std::sort(sim_trk_ihits.begin(), sim_trk_ihits.end());
    std::sort(hit_idxs.begin(), hit_idxs.end());

    std::vector<unsigned int> v_intersection;

    std::set_intersection(sim_trk_ihits.begin(), sim_trk_ihits.end(),
                          hit_idxs.begin(), hit_idxs.end(),
                          std::back_inserter(v_intersection));

    if (verbose)
    {
        if (static_cast<int>(v_intersection.size()) > ana.nmatch_threshold)
        {
            std::cout << "Matched" << std::endl;
        }
        else
        {
            std::cout << "Not matched" << std::endl;
        }
        std::cout << "sim_trk_ihits: ";
        for (auto& i_simhit_idx : sim_trk_ihits)
            std::cout << i_simhit_idx << " ";
        std::cout << std::endl;

        std::cout << "     hit_idxs: ";
        for (auto& i_hit_idx : hit_idxs)
            std::cout << i_hit_idx << " ";
        std::cout << std::endl;
    }

    int nhits = hit_idxs.size();

    float factor = nhits / 12.;

    // If 75% of 12 hits have been found than it is matched
    return (v_intersection.size() > ana.nmatch_threshold * factor);
}

void loadMaps()
{
    TString TrackLooperDir = gSystem->Getenv("TRACKLOOPERDIR");

#ifdef CMSSW12GEOM
    std::cout << "Loading CMSSW_12_2_0_pre2 geometry" << std::endl;
#else
    std::cout << "Loading MTD TDR geometry" << std::endl;
#endif

    // Module orientation information (DrDz or phi angles)
#ifdef CMSSW12GEOM
    TString endcap_geom = get_absolute_path_after_check_file_exists(TString::Format("%s/data/endcap_orientation_data_CMSSW_12_2_0_pre2.txt", TrackLooperDir.Data()).Data());
    TString tilted_geom = get_absolute_path_after_check_file_exists(TString::Format("%s/data/tilted_orientation_data_CMSSW_12_2_0_pre2.txt", TrackLooperDir.Data()).Data());
#else
    TString endcap_geom = get_absolute_path_after_check_file_exists(TString::Format("%s/data/endcap_orientation_data_v2.txt", TrackLooperDir.Data()).Data()); // centroid values added to the map
    TString tilted_geom = get_absolute_path_after_check_file_exists(TString::Format("%s/data/tilted_orientation_data.txt", TrackLooperDir.Data()).Data());
#endif
    std::cout << "Loading module orientation information...." << std::endl;
    std::cout << "endcap orientation:" << endcap_geom << std::endl;
    std::cout << "tilted orientation:" << tilted_geom << std::endl;
    SDL::endcapGeometry.load(endcap_geom.Data()); // centroid values added to the map
    SDL::tiltedGeometry.load(tilted_geom.Data());

    // Module connection map (for line segment building)
#ifdef PT0P8
#ifdef CMSSW12GEOM
    TString mappath = get_absolute_path_after_check_file_exists(TString::Format("%s/data/module_connection_tracing_CMSSW_12_2_0_pre2_merged.txt", TrackLooperDir.Data()).Data());
#else
    TString mappath = get_absolute_path_after_check_file_exists(TString::Format("%s/data/module_connection_combined_0p8helix_muongun.txt", TrackLooperDir.Data()).Data());
#endif
#else
#ifdef CMSSW12GEOM
    // TODO: The CMSSW_12_2_0_pre2 is by default 0.8 module map
    TString mappath = get_absolute_path_after_check_file_exists(TString::Format("%s/data/module_connection_tracing_CMSSW_12_2_0_pre2_merged.txt", TrackLooperDir.Data()).Data());
#else
    TString mappath = get_absolute_path_after_check_file_exists(TString::Format("%s/data/module_connection_combined_2020_0520_helixray.txt", TrackLooperDir.Data()).Data());
#endif
#endif
    std::cout << "Loading module map...." << std::endl;
    std::cout << "module map path:" << mappath << std::endl;
    SDL::moduleConnectionMap.load(mappath.Data());
    ana.moduleConnectiongMapLoose.load(mappath.Data());

#ifdef PT0P8
#ifdef CMSSW12GEOM
    TString pLSMapDir = "/data2/segmentlinking/pixelmap_CMSSW_12_2_0_pre2_0p8minPt"; // baseline + 0.8 GeV
#else
    TString pLSMapDir = "/data2/segmentlinking/pixelmap_ptmin0p8_neta25_nphi72_nz25_ipt2_etapm0p05_zpm0p05"; // baseline + 0.8 GeV
#endif
#else
#ifdef CMSSW12GEOM
    // TODO: The CMSSW_12_2_0_pre2 is by default 0.8 module map
    TString pLSMapDir = "/data2/segmentlinking/pixelmap_CMSSW_12_2_0_pre2_0p8minPt"; // baseline
#else
    TString pLSMapDir = "/data2/segmentlinking/pixelmap_neta25_nphi72_nz25_ipt2_etapm0p05_zpm0p05"; // baseline
#endif
#endif

    std::cout << "Loading pLS maps ... from pLSMapDir = " << pLSMapDir << std::endl;

    TString path;
    path = TString::Format("%s/pLS_map_layer1_subdet5.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer1Subdet5.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
    path = TString::Format("%s/pLS_map_layer2_subdet5.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer2Subdet5.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
    path = TString::Format("%s/pLS_map_layer3_subdet5.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer3Subdet5.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
    path = TString::Format("%s/pLS_map_layer1_subdet4.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer1Subdet4.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
    path = TString::Format("%s/pLS_map_layer2_subdet4.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer2Subdet4.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
    path = TString::Format("%s/pLS_map_layer3_subdet4.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer3Subdet4.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
    path = TString::Format("%s/pLS_map_layer4_subdet4.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer4Subdet4.load(get_absolute_path_after_check_file_exists(path.Data()).Data());

    path = TString::Format("%s/pLS_map_neg_layer1_subdet5.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer1Subdet5_neg.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
    path = TString::Format("%s/pLS_map_neg_layer2_subdet5.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer2Subdet5_neg.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
    path = TString::Format("%s/pLS_map_neg_layer3_subdet5.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer3Subdet5_neg.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
    path = TString::Format("%s/pLS_map_neg_layer1_subdet4.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer1Subdet4_neg.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
    path = TString::Format("%s/pLS_map_neg_layer2_subdet4.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer2Subdet4_neg.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
    path = TString::Format("%s/pLS_map_neg_layer3_subdet4.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer3Subdet4_neg.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
    path = TString::Format("%s/pLS_map_neg_layer4_subdet4.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer4Subdet4_neg.load(get_absolute_path_after_check_file_exists(path.Data()).Data());

    path = TString::Format("%s/pLS_map_pos_layer1_subdet5.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer1Subdet5_pos.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
    path = TString::Format("%s/pLS_map_pos_layer2_subdet5.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer2Subdet5_pos.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
    path = TString::Format("%s/pLS_map_pos_layer3_subdet5.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer3Subdet5_pos.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
    path = TString::Format("%s/pLS_map_pos_layer1_subdet4.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer1Subdet4_pos.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
    path = TString::Format("%s/pLS_map_pos_layer2_subdet4.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer2Subdet4_pos.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
    path = TString::Format("%s/pLS_map_pos_layer3_subdet4.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer3Subdet4_pos.load(get_absolute_path_after_check_file_exists(path.Data()).Data());
    path = TString::Format("%s/pLS_map_pos_layer4_subdet4.txt", pLSMapDir.Data()).Data(); SDL::moduleConnectionMap_pLStoLayer4Subdet4_pos.load(get_absolute_path_after_check_file_exists(path.Data()).Data());

    // SDL::moduleConnectionMap.load("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/scripts/module_connection_map_data_10_e0_200_100_pt0p8_2p0_400_pt0p8_2p0_nolossers_dxy35cm_endcaplayer2.txt");
    // ana.moduleConnectiongMapLoose.load("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/scripts/module_connection_map_data_10_e0_200_100_pt0p8_2p0_400_pt0p8_2p0_nolossers_dxy35cm_endcaplayer2.txt");

    // SDL::moduleConnectionMap.load("data/module_connection_2020_0429.txt");
    // ana.moduleConnectiongMapLoose.load("data/module_connection_2020_0429.txt");

    // SDL::moduleConnectionMap.load("data/module_connection_tracing_2020_0514.txt");
    // SDL::moduleConnectionMap.load("data/module_connection_combined_2020_0518_helixray.txt");

}


float drfracSimHitConsistentWithHelix(int isimtrk, int isimhitidx)
{
    // Read track parameters
    float vx = trk.simvtx_x()[0];
    float vy = trk.simvtx_y()[0];
    float vz = trk.simvtx_z()[0];
    float pt = trk.sim_pt()[isimtrk];
    float eta = trk.sim_eta()[isimtrk];
    float phi = trk.sim_phi()[isimtrk];
    float charge = trk.sim_q()[isimtrk];

    // Construct helix object
    SDLMath::Helix helix(pt, eta, phi, vx, vy, vz, charge);

    return drfracSimHitConsistentWithHelix(helix, isimhitidx);

}

//__________________________________________________________________________________________
float drfracSimHitConsistentWithHelix(SDLMath::Helix& helix, int isimhitidx)
{

    // Sim hit vector
    std::vector<float> point = {trk.simhit_x()[isimhitidx], trk.simhit_y()[isimhitidx], trk.simhit_z()[isimhitidx]};

    // Inferring parameter t of helix parametric function via z position
    float t = helix.infer_t(point);

    // If the best fit is more than pi parameter away then it's a returning hit and should be ignored
    if (not (t <= M_PI))
        return 999;

    // Expected hit position with given z
    auto [x, y, z, r] = helix.get_helix_point(t);

    // ( expected_r - simhit_r ) / expected_r
    float drfrac = abs(helix.compare_radius(point)) / r;

    return drfrac;

}

//__________________________________________________________________________________________
float distxySimHitConsistentWithHelix(int isimtrk, int isimhitidx)
{
    // Read track parameters
    float vx = trk.simvtx_x()[0];
    float vy = trk.simvtx_y()[0];
    float vz = trk.simvtx_z()[0];
    float pt = trk.sim_pt()[isimtrk];
    float eta = trk.sim_eta()[isimtrk];
    float phi = trk.sim_phi()[isimtrk];
    float charge = trk.sim_q()[isimtrk];

    // Construct helix object
    SDLMath::Helix helix(pt, eta, phi, vx, vy, vz, charge);

    return distxySimHitConsistentWithHelix(helix, isimhitidx);

}

//__________________________________________________________________________________________
float distxySimHitConsistentWithHelix(SDLMath::Helix& helix, int isimhitidx)
{

    // Sim hit vector
    std::vector<float> point = {trk.simhit_x()[isimhitidx], trk.simhit_y()[isimhitidx], trk.simhit_z()[isimhitidx]};

    // Inferring parameter t of helix parametric function via z position
    float t = helix.infer_t(point);

    // If the best fit is more than pi parameter away then it's a returning hit and should be ignored
    if (not (t <= M_PI))
        return 999;

    // Expected hit position with given z
    //auto [x, y, z, r] = helix.get_helix_point(t);

    // ( expected_r - simhit_r ) / expected_r
    float distxy = helix.compare_xy(point);

    return distxy;

}

//__________________________________________________________________________________________
void addInputsToLineSegmentTrackingPreLoad(
std::vector<std::vector<float>>& out_trkX,std::vector<std::vector<float>>& out_trkY,std::vector<std::vector<float>>& out_trkZ,

std::vector<std::vector<unsigned int>>&    out_hitId,
std::vector<std::vector<unsigned int>>&    out_hitIdxs,
std::vector<std::vector<unsigned int>>&    out_hitIndices_vec0,
std::vector<std::vector<unsigned int>>&    out_hitIndices_vec1,
std::vector<std::vector<unsigned int>>&    out_hitIndices_vec2,
std::vector<std::vector<unsigned int>>&    out_hitIndices_vec3,
std::vector<std::vector<float>>&    out_deltaPhi_vec,
std::vector<std::vector<float>>&    out_ptIn_vec,
std::vector<std::vector<float>>&    out_ptErr_vec,
std::vector<std::vector<float>>&    out_px_vec,
std::vector<std::vector<float>>&    out_py_vec,
std::vector<std::vector<float>>&    out_pz_vec,
std::vector<std::vector<float>>&    out_eta_vec,
std::vector<std::vector<float>>&    out_etaErr_vec,
std::vector<std::vector<float>>&    out_phi_vec,
std::vector<std::vector<float>>&    out_charge_vec,
std::vector<std::vector<int>>&    out_superbin_vec,
std::vector<std::vector<int8_t>>&    out_pixelType_vec,
std::vector<std::vector<short>>&    out_isQuad_vec)
{

//    TStopwatch my_timer;
//    if (ana.verbose >= 2) std::cout << "Loading Inputs (i.e. outer tracker hits, and pixel line segements) to the Line Segment Tracking.... " << std::endl;
//    my_timer.Start();

    unsigned int count = 0;
    auto n_see = trk.see_stateTrajGlbPx().size();
    std::vector<float> px_vec; px_vec.reserve(n_see);
    std::vector<float> py_vec; py_vec.reserve(n_see);
    std::vector<float> pz_vec; pz_vec.reserve(n_see);
    std::vector<unsigned int> hitIndices_vec0; hitIndices_vec0.reserve(n_see);
    std::vector<unsigned int> hitIndices_vec1; hitIndices_vec1.reserve(n_see);
    std::vector<unsigned int> hitIndices_vec2; hitIndices_vec2.reserve(n_see);
    std::vector<unsigned int> hitIndices_vec3; hitIndices_vec3.reserve(n_see);
    std::vector<float> ptIn_vec;   ptIn_vec.reserve(n_see);
    std::vector<float> ptErr_vec;  ptErr_vec.reserve(n_see);
    std::vector<float> etaErr_vec; etaErr_vec.reserve(n_see);
    std::vector<float> eta_vec;    eta_vec.reserve(n_see);
    std::vector<float> phi_vec;    phi_vec.reserve(n_see);
    std::vector<float> charge_vec;    charge_vec.reserve(n_see);
    std::vector<float> deltaPhi_vec; deltaPhi_vec.reserve(n_see);
    std::vector<float> trkX = trk.ph2_x();
    std::vector<float> trkY = trk.ph2_y();
    std::vector<float> trkZ = trk.ph2_z();
    std::vector<unsigned int> hitId = trk.ph2_detId();
    std::vector<unsigned int> hitIdxs(trk.ph2_detId().size());

    std::vector<int> superbin_vec;
    std::vector<int8_t> pixelType_vec;
    std::vector<short> isQuad_vec;
    std::iota(hitIdxs.begin(), hitIdxs.end(), 0);
    const int hit_size = trkX.size();

    for (auto &&[iSeed, _] : iter::enumerate(trk.see_stateTrajGlbPx()))
    {
        bool good_seed_type = false;
        if (trk.see_algo()[iSeed] == 4) good_seed_type = true;
        if (trk.see_algo()[iSeed] == 22) good_seed_type = true;
        if (not good_seed_type) continue;

        TVector3 p3LH(trk.see_stateTrajGlbPx()[iSeed], trk.see_stateTrajGlbPy()[iSeed], trk.see_stateTrajGlbPz()[iSeed]);
        float ptIn = p3LH.Pt();
        float eta = p3LH.Eta();
        float ptErr = trk.see_ptErr()[iSeed];

#ifdef PT0P8
        if ((ptIn > 0.8 - 2 * ptErr))
#else
        if ((ptIn > 1 - 2 * ptErr) and (fabs(eta) < 3))
#endif
        {
        TVector3 r3LH(trk.see_stateTrajGlbX()[iSeed], trk.see_stateTrajGlbY()[iSeed], trk.see_stateTrajGlbZ()[iSeed]);
        TVector3 p3PCA(trk.see_px()[iSeed], trk.see_py()[iSeed], trk.see_pz()[iSeed]);
        TVector3 r3PCA(calculateR3FromPCA(p3PCA, trk.see_dxy()[iSeed], trk.see_dz()[iSeed]));
        TVector3 seedSD_mdRef_r3 = r3PCA;
        TVector3 seedSD_mdOut_r3 = r3LH;
        TVector3 seedSD_r3 = r3LH;
        TVector3 seedSD_p3 = p3LH;

        float pixelSegmentDeltaPhiChange = r3LH.DeltaPhi(p3LH);
        float etaErr = trk.see_etaErr()[iSeed];
        float px = p3LH.X();
        float py = p3LH.Y();
        float pz = p3LH.Z();

        float charge = trk.see_q()[iSeed];
        //extra bit
            // get pixel superbin
            //int ptbin = -1;
            int pixtype =-1;

            if (ptIn >= 2.0){ /*ptbin = 1;*/pixtype=0;}
            else if (ptIn >= (0.8 - 2 * ptErr) and ptIn < 2.0){ 
              //ptbin = 0;
              if (pixelSegmentDeltaPhiChange >= 0){pixtype=1;}
              else{pixtype=2;}
            }
            else{continue;}

// all continues before pushing back into vectots to avoid strange offsets in indicies. 
            unsigned int hitIdx0 = hit_size + count;
            count++; 

            unsigned int hitIdx1 = hit_size + count;
            count++;

            unsigned int hitIdx2 = hit_size + count;
            count++;

            unsigned int hitIdx3;
            if (trk.see_hitIdx()[iSeed].size() <= 3)
            {
                hitIdx3 = hitIdx2;
            }
            else
            {
                hitIdx3 = hit_size + count;
                count++;
            }

            trkX.push_back(r3PCA.X());
            trkY.push_back(r3PCA.Y());
            trkZ.push_back(r3PCA.Z());
            trkX.push_back(p3PCA.Pt());
            float p3PCA_Eta = p3PCA.Eta();
            trkY.push_back(p3PCA_Eta);
            float p3PCA_Phi = p3PCA.Phi();
            trkZ.push_back(p3PCA_Phi);
            trkX.push_back(r3LH.X());
            trkY.push_back(r3LH.Y());
            trkZ.push_back(r3LH.Z());
            hitId.push_back(1);
            hitId.push_back(1);
            hitId.push_back(1);
            if(trk.see_hitIdx()[iSeed].size() > 3)
            {
                trkX.push_back(r3LH.X());
                trkY.push_back(trk.see_dxy()[iSeed]);
                trkZ.push_back(trk.see_dz()[iSeed]);
                hitId.push_back(1);
            }
            px_vec.push_back(px);
            py_vec.push_back(py);
            pz_vec.push_back(pz);

            hitIndices_vec0.push_back(hitIdx0);
            hitIndices_vec1.push_back(hitIdx1);
            hitIndices_vec2.push_back(hitIdx2);
            hitIndices_vec3.push_back(hitIdx3);
            ptIn_vec.push_back(ptIn);
            ptErr_vec.push_back(ptErr);
            etaErr_vec.push_back(etaErr);
            eta_vec.push_back(eta);
            float phi = p3LH.Phi();
            phi_vec.push_back(phi);
            charge_vec.push_back(charge);
            deltaPhi_vec.push_back(pixelSegmentDeltaPhiChange);

            // For matching with sim tracks
            hitIdxs.push_back(trk.see_hitIdx()[iSeed][0]);
            hitIdxs.push_back(trk.see_hitIdx()[iSeed][1]);
            hitIdxs.push_back(trk.see_hitIdx()[iSeed][2]);
            bool isQuad = false;
            if(trk.see_hitIdx()[iSeed].size() > 3)
            {
                isQuad = true;
                hitIdxs.push_back(trk.see_hitIdx()[iSeed].size() > 3 ? trk.see_hitIdx()[iSeed][3] : trk.see_hitIdx()[iSeed][2]);
            }
            //if (pt < 0){ ptbin = 0;}
            float neta = 25.;
            float nphi = 72.;
            float nz = 25.;
            int etabin = (p3PCA_Eta + 2.6) / ((2*2.6)/neta);
            int phibin = (p3PCA_Phi + 3.14159265358979323846) / ((2.*3.14159265358979323846) / nphi);
            int dzbin = (trk.see_dz()[iSeed] + 30) / (2*30 / nz);
            int isuperbin = /*(nz * nphi * neta) * ptbin + (removed since pt bin is determined by pixelType)*/ (nz * nphi) * etabin + (nz) * phibin + dzbin;
            superbin_vec.push_back(isuperbin);
            pixelType_vec.push_back(pixtype);
            isQuad_vec.push_back(isQuad);

        }

    } // iter::enumerate(trk.see_stateTrajGlbPx

//    event.addHitToEvent(trkX, trkY, trkZ, hitId,hitIdxs); // TODO : Need to fix the hitIdxs
//    event.addPixelSegmentToEvent(hitIndices_vec0, hitIndices_vec1, hitIndices_vec2, hitIndices_vec3, deltaPhi_vec, ptIn_vec, ptErr_vec, px_vec, py_vec, pz_vec, eta_vec, etaErr_vec, phi_vec, charge_vec, superbin_vec, pixelType_vec,isQuad_vec);

    out_trkX.push_back(trkX);
    out_trkY.push_back(trkY);
    out_trkZ.push_back(trkZ);
    out_hitId.push_back(hitId);
    out_hitIdxs.push_back(hitIdxs);
    out_hitIndices_vec0.push_back(hitIndices_vec0);
    out_hitIndices_vec1.push_back(hitIndices_vec1);
    out_hitIndices_vec2.push_back(hitIndices_vec2);
    out_hitIndices_vec3.push_back(hitIndices_vec3);
    out_deltaPhi_vec.push_back(deltaPhi_vec);
    out_ptIn_vec.push_back(ptIn_vec);
    out_ptErr_vec.push_back(ptErr_vec);
    out_px_vec.push_back(px_vec);
    out_py_vec.push_back(py_vec);
    out_pz_vec.push_back(pz_vec);
    out_eta_vec.push_back(eta_vec);
    out_etaErr_vec.push_back(etaErr_vec);
    out_phi_vec.push_back(phi_vec);
    out_charge_vec.push_back(charge_vec);
    out_superbin_vec.push_back(superbin_vec);
    out_pixelType_vec.push_back(pixelType_vec);
    out_isQuad_vec.push_back(isQuad_vec);
    
}

//float addInputsToEventPreLoad(SDL::Event& event, bool useOMP,
float addInputsToEventPreLoad(SDL::Event* event, bool useOMP,
std::vector<float> trkX,std::vector<float> trkY,std::vector<float> trkZ,
std::vector<unsigned int>    hitId,
std::vector<unsigned int>    hitIdxs,
std::vector<unsigned int>    hitIndices_vec0,
std::vector<unsigned int>    hitIndices_vec1,
std::vector<unsigned int>    hitIndices_vec2,
std::vector<unsigned int>    hitIndices_vec3,
std::vector<float>    deltaPhi_vec,
std::vector<float>    ptIn_vec,
std::vector<float>    ptErr_vec,
std::vector<float>    px_vec,
std::vector<float>    py_vec,
std::vector<float>    pz_vec,
std::vector<float>    eta_vec,
std::vector<float>    etaErr_vec,
std::vector<float>    phi_vec,
std::vector<float>    charge_vec,
std::vector<int>    superbin_vec,
std::vector<int8_t>    pixelType_vec,
std::vector<short>    isQuad_vec)
{
    TStopwatch my_timer;
    if (ana.verbose >= 2) std::cout << "Loading Inputs (i.e. outer tracker hits, and pixel line segements) to the Line Segment Tracking.... " << std::endl;
    my_timer.Start();
    event->addHitToEvent(trkX, trkY, trkZ, hitId,hitIdxs); // TODO : Need to fix the hitIdxs
    event->addPixelSegmentToEvent(hitIndices_vec0, hitIndices_vec1, hitIndices_vec2, hitIndices_vec3, deltaPhi_vec, ptIn_vec, ptErr_vec, px_vec, py_vec, pz_vec, eta_vec, etaErr_vec, phi_vec, charge_vec, superbin_vec, pixelType_vec,isQuad_vec);
    float hit_loading_elapsed = my_timer.RealTime();
    if (ana.verbose >= 2) std::cout << "Loading inputs processing time: " << hit_loading_elapsed << " secs" << std::endl;
    return hit_loading_elapsed;
}

//__________________________________________________________________________________________
float addInputsToLineSegmentTracking(SDL::Event &event, bool useOMP)
{

    TStopwatch my_timer;
    if (ana.verbose >= 2) std::cout << "Loading Inputs (i.e. outer tracker hits, and pixel line segements) to the Line Segment Tracking.... " << std::endl;
    my_timer.Start();

    unsigned int count = 0;
    std::vector<float> px_vec;
    std::vector<float> py_vec;
    std::vector<float> pz_vec;
    std::vector<unsigned int> hitIndices_vec0;
    std::vector<unsigned int> hitIndices_vec1;
    std::vector<unsigned int> hitIndices_vec2;
    std::vector<unsigned int> hitIndices_vec3;
    std::vector<float> ptIn_vec;
    std::vector<float> ptErr_vec;
    std::vector<float> etaErr_vec;
    std::vector<float> eta_vec;
    std::vector<float> phi_vec; 
    std::vector<float> charge_vec;
    std::vector<float> deltaPhi_vec;
    std::vector<float> trkX = trk.ph2_x();
    std::vector<float> trkY = trk.ph2_y();
    std::vector<float> trkZ = trk.ph2_z();
    std::vector<unsigned int> hitId = trk.ph2_detId();
    std::vector<unsigned int> hitIdxs(trk.ph2_detId().size());
    std::vector<int> superbin_vec;
    std::vector<int8_t> pixelType_vec;
    std::vector<short> isQuad_vec;
    std::iota(hitIdxs.begin(), hitIdxs.end(), 0);
    const int hit_size = trkX.size();

    for (auto &&[iSeed, _] : iter::enumerate(trk.see_stateTrajGlbPx()))
    {
        bool good_seed_type = false;
        if (trk.see_algo()[iSeed] == 4) good_seed_type = true;
        if (trk.see_algo()[iSeed] == 22) good_seed_type = true;
        if (not good_seed_type) continue;

        TVector3 p3LH(trk.see_stateTrajGlbPx()[iSeed], trk.see_stateTrajGlbPy()[iSeed], trk.see_stateTrajGlbPz()[iSeed]);
        float ptIn = p3LH.Pt();
        float ptErr = trk.see_ptErr()[iSeed];
        float eta = p3LH.Eta();

#ifdef PT0P8
        if ((ptIn > 0.8 - 2 * ptErr))
#else
        if ((ptIn > 1 - 2 * ptErr) and (fabs(eta) < 3))
#endif
        {
        TVector3 r3LH(trk.see_stateTrajGlbX()[iSeed], trk.see_stateTrajGlbY()[iSeed], trk.see_stateTrajGlbZ()[iSeed]);
        TVector3 p3PCA(trk.see_px()[iSeed], trk.see_py()[iSeed], trk.see_pz()[iSeed]);
        TVector3 r3PCA(calculateR3FromPCA(p3PCA, trk.see_dxy()[iSeed], trk.see_dz()[iSeed]));

        TVector3 seedSD_mdRef_r3 = r3PCA;
        TVector3 seedSD_mdOut_r3 = r3LH;
        TVector3 seedSD_r3 = r3LH;
        TVector3 seedSD_p3 = p3LH;

        float pixelSegmentDeltaPhiChange = r3LH.DeltaPhi(p3LH);
        float etaErr = trk.see_etaErr()[iSeed];
        float px = p3LH.X();
        float py = p3LH.Y();
        float pz = p3LH.Z();
        float phi = p3LH.Phi();
        float charge = trk.see_q()[iSeed];
        //extra bit
	
            // get pixel superbin
            //int ptbin = -1;
            int pixtype =-1;
            if (ptIn >= 2.0){ /*ptbin = 1;*/pixtype=0;}
            else if (ptIn >= (0.8 - 2 * ptErr) and ptIn < 2.0){ 
              //ptbin = 0;
              if (pixelSegmentDeltaPhiChange >= 0){pixtype=1;}
              else{pixtype=2;}
            }
            else{continue;}

// all continues before pushing back into vectots to avoid strange offsets in indicies. 
            unsigned int hitIdx0 = hit_size + count;
            count++; 

            unsigned int hitIdx1 = hit_size + count;
            count++;

            unsigned int hitIdx2 = hit_size + count;
            count++;

            unsigned int hitIdx3;
            if (trk.see_hitIdx()[iSeed].size() <= 3)
            {
                hitIdx3 = hitIdx2;
            }
            else
            {
                hitIdx3 = hit_size + count;
                count++;
            }

            trkX.push_back(r3PCA.X());
            trkY.push_back(r3PCA.Y());
            trkZ.push_back(r3PCA.Z());
            trkX.push_back(p3PCA.Pt());
            float p3PCA_Eta = p3PCA.Eta();
            trkY.push_back(p3PCA_Eta);
            float p3PCA_Phi = p3PCA.Phi();
            trkZ.push_back(p3PCA_Phi);
            trkX.push_back(r3LH.X());
            trkY.push_back(r3LH.Y());
            trkZ.push_back(r3LH.Z());
            hitId.push_back(1);
            hitId.push_back(1);
            hitId.push_back(1);
            if(trk.see_hitIdx()[iSeed].size() > 3)
            {
                trkX.push_back(r3LH.X());
                trkY.push_back(trk.see_dxy()[iSeed]);
                trkZ.push_back(trk.see_dz()[iSeed]);
                hitId.push_back(1);
            }
            px_vec.push_back(px);
            py_vec.push_back(py);
            pz_vec.push_back(pz);

            hitIndices_vec0.push_back(hitIdx0);
            hitIndices_vec1.push_back(hitIdx1);
            hitIndices_vec2.push_back(hitIdx2);
            hitIndices_vec3.push_back(hitIdx3);
            ptIn_vec.push_back(ptIn);
            ptErr_vec.push_back(ptErr);
            etaErr_vec.push_back(etaErr);
            eta_vec.push_back(eta);
            phi_vec.push_back(phi);
            charge_vec.push_back(charge);
            deltaPhi_vec.push_back(pixelSegmentDeltaPhiChange);

            // For matching with sim tracks
            hitIdxs.push_back(trk.see_hitIdx()[iSeed][0]);
            hitIdxs.push_back(trk.see_hitIdx()[iSeed][1]);
            hitIdxs.push_back(trk.see_hitIdx()[iSeed][2]);
            bool isQuad = false;
            if(trk.see_hitIdx()[iSeed].size() > 3)
            {
                isQuad = true;
                hitIdxs.push_back(trk.see_hitIdx()[iSeed].size() > 3 ? trk.see_hitIdx()[iSeed][3] : trk.see_hitIdx()[iSeed][2]);
            }
            //if (pt < 0){ ptbin = 0;}
            float neta = 25.;
            float nphi = 72.;
            float nz = 25.;
            int etabin = (p3PCA_Eta + 2.6) / ((2*2.6)/neta);
            int phibin = (p3PCA_Phi + 3.14159265358979323846) / ((2.*3.14159265358979323846) / nphi);
            int dzbin = (trk.see_dz()[iSeed] + 30) / (2*30 / nz);
            int isuperbin = /*(nz * nphi * neta) * ptbin + (removed since pt bin is determined by pixelType)*/ (nz * nphi) * etabin + (nz) * phibin + dzbin;
            superbin_vec.push_back(isuperbin);
            pixelType_vec.push_back(pixtype);
            isQuad_vec.push_back(isQuad);
        }
    }

    event.addHitToEvent(trkX, trkY, trkZ, hitId,hitIdxs); // TODO : Need to fix the hitIdxs
    event.addPixelSegmentToEvent(hitIndices_vec0, hitIndices_vec1, hitIndices_vec2, hitIndices_vec3, deltaPhi_vec, ptIn_vec, ptErr_vec, px_vec, py_vec, pz_vec, eta_vec, etaErr_vec, phi_vec, charge_vec, superbin_vec, pixelType_vec,isQuad_vec);

    float hit_loading_elapsed = my_timer.RealTime();
    if (ana.verbose >= 2) std::cout << "Loading inputs processing time: " << hit_loading_elapsed << " secs" << std::endl;
    return hit_loading_elapsed;
}


//__________________________________________________________________________________________
float addInputsToLineSegmentTrackingUsingExplicitMemory(SDL::Event &event)
{
    return addInputsToLineSegmentTracking(event, true);
}

//__________________________________________________________________________________________
TVector3 calculateR3FromPCA(const TVector3& p3, const float dxy, const float dz)
{
  const float pt = p3.Pt();
  const float p = p3.Mag();
  const float vz = dz*pt*pt/p/p;

  const float vx = -dxy*p3.y()/pt - p3.x()/p*p3.z()/p*dz;
  const float vy =  dxy*p3.x()/pt - p3.y()/p*p3.z()/p*dz;
  return TVector3(vx, vy, vz);
}

//__________________________________________________________________________________________
//float addPixelSegments(SDL::CPU::Event& event, int isimtrk)
//{
//
//    TStopwatch my_timer;
//    if (ana.verbose >= 2) std::cout << "Loading pixel line segments for CPU...." << std::endl;
//    my_timer.Start();
//
//    for (auto&& [iSeed, _] : iter::enumerate(trk.see_stateTrajGlbPx()))
//    {
//
//        if (isimtrk >= 0)
//        {
//            bool match = false;
//            for (auto& seed_simtrkidx : trk.see_simTrkIdx()[iSeed])
//            {
//                if (seed_simtrkidx == isimtrk)
//                {
//                    match = true;
//                }
//            }
//            if (not match)
//                continue;
//        }
//
//        if (abs(ana.pdg_id) != 0)
//        {
//            bool match = false;
//            for (auto& seed_simtrkidx : trk.see_simTrkIdx()[iSeed])
//            {
//                if (abs(trk.sim_pdgId()[seed_simtrkidx]) == ana.pdg_id)
//                {
//                    match = true;
//                }
//            }
//            if (not match)
//                continue;
//        }
//
//        TVector3 p3LH(trk.see_stateTrajGlbPx()[iSeed], trk.see_stateTrajGlbPy()[iSeed], trk.see_stateTrajGlbPz()[iSeed]);
//        TVector3 r3LH(trk.see_stateTrajGlbX()[iSeed], trk.see_stateTrajGlbY()[iSeed], trk.see_stateTrajGlbZ()[iSeed]);
//        TVector3 p3PCA(trk.see_px()[iSeed], trk.see_py()[iSeed], trk.see_pz()[iSeed]);
//        TVector3 r3PCA(calculateR3FromPCA(p3PCA, trk.see_dxy()[iSeed], trk.see_dz()[iSeed]));
//        //auto const& seedHitsV = trk.see_hitIdx()[iSeed];
//        //auto const& seedHitTypesV = trk.see_hitType()[iSeed];
//
//        // /// track algorithm; partial copy from TrackBase.h
//        // enum class TrackAlgorithm {
//        //     undefAlgorithm = 0,
//        //     ctf = 1,
//        //     duplicateMerge = 2,
//        //     cosmics = 3,
//        //     initialStep = 4,
//        //     lowPtTripletStep = 5,
//        //     pixelPairStep = 6,
//        //     detachedTripletStep = 7,
//        //     mixedTripletStep = 8,
//        //     pixelLessStep = 9,
//        //     tobTecStep = 10,
//        //     jetCoreRegionalStep = 11,
//        //     conversionStep = 12,
//        //     muonSeededStepInOut = 13,
//        //     muonSeededStepOutIn = 14,
//        //     outInEcalSeededConv = 15, inOutEcalSeededConv = 16,
//        //     nuclInter = 17,
//        //     standAloneMuon = 18, globalMuon = 19, cosmicStandAloneMuon = 20, cosmicGlobalMuon = 21,
//        //     // Phase1
//        //     highPtTripletStep = 22, lowPtQuadStep = 23, detachedQuadStep = 24,
//        //     reservedForUpgrades1 = 25, reservedForUpgrades2 = 26,
//        //     bTagGhostTracks = 27,
//        //     beamhalo = 28,
//        //     gsf = 29
//        // };
//        bool good_seed_type = false;
//        if (trk.see_algo()[iSeed] == 4) good_seed_type = true;
//        if (trk.see_algo()[iSeed] == 5) good_seed_type = true;
//        if (trk.see_algo()[iSeed] == 7) good_seed_type = true;
//        if (trk.see_algo()[iSeed] == 22) good_seed_type = true;
//        if (trk.see_algo()[iSeed] == 23) good_seed_type = true;
//        if (trk.see_algo()[iSeed] == 24) good_seed_type = true;
//        if (not good_seed_type)
//            continue;
//        // if (trk.see_algo()[iSeed] != 4 and trk.see_algo()[iSeed] != 5)
//        //     continue;
//        // if (trk.see_algo()[iSeed] < 4 and trk.see_algo()[iSeed] > 8)
//        //     continue;
//
//        //int nHits = seedHitsV.size();
//
//        //assert(nHits == 4);
//        //for (int iH = 0; iH < nHits; ++iH){
//        //    //FIXME: make this configurable
//        //    assert(seedHitTypesV[iH] == 0);
//        //}
//
//        // float seedSD_mdRef_pixL = HitIndexWithType(trk.see_hitIdx()[iSeed][0], HitType(trk.see_hitType()[iSeed][0])).indexWithType;
//        // float seedSD_mdRef_pixU = HitIndexWithType(trk.see_hitIdx()[iSeed][1], HitType(trk.see_hitType()[iSeed][1])).indexWithType;
//        //int seedSD_mdRef_pixL = trk.see_hitIdx()[iSeed][0];
//        //int seedSD_mdRef_pixU = trk.see_hitIdx()[iSeed][1];
//        TVector3 seedSD_mdRef_r3 = r3PCA;
//        //float seedSD_mdRef_rt = r3PCA.Pt();
//        //float seedSD_mdRef_z = r3PCA.Z();
//        //float seedSD_mdRef_r = r3PCA.Mag();
//        //float seedSD_mdRef_phi = r3PCA.Phi();
//        //float seedSD_mdRef_alpha = r3PCA.DeltaPhi(p3PCA);
//        // const int itpRL = simsPerHitAll(seedSD_mdRef_pixL); // TODO: best sim trk idx SO PERHAPS NOT NEEDED
//        // const int itpRU = simsPerHitAll(seedSD_mdRef_pixU); // TODO: best sim trk idx
//        // float seedSD_mdRef_itp = itpRL;
//        // float seedSD_mdRef_ntp = 1;
//        // float seedSD_mdRef_itpLL = itpRL;
//
//        // if (itpRL >= 0 && itpRU >= 0)
//        // {
//        //     if (itpRL == itpRU)
//        //     {
//        //         seedSD_mdRef_ntp = 2;
//        //     }
//        // }
//        // else if (itpRL == -1 && itpRU == -1)
//        // {
//        //     seedSD_mdRef_ntp = 0;
//        // }
//        // else if (itpRU >= 0)
//        // {
//        //     seedSD_mdRef_itp = itpRU;
//        // }
//
//        //int seedSD_mdOut_pixL = trk.see_hitIdx()[iSeed][2];
//        //int seedSD_mdOut_pixU = trk.see_hitIdx()[iSeed][3];
//        // if (nPix >= 4)
//        //     seedSD_mdOut_pixU = trk.see_hitIdx()[iSeed][3];
//        TVector3 seedSD_mdOut_r3 = r3LH;
//        //float seedSD_mdOut_rt = r3LH.Pt();
//        //float seedSD_mdOut_z = r3LH.Z();
//        //float seedSD_mdOut_r = r3LH.Mag();
//        //float seedSD_mdOut_phi = r3LH.Phi();
//        //float seedSD_mdOut_alpha = r3LH.DeltaPhi(p3LH);
//        // const int itpOL = simsPerHitAll(seedSD_mdOut_pixL);
//        // const int itpOU = simsPerHitAll(seedSD_mdOut_pixU);
//        // float seedSD_mdOut_itp = itpOL;
//        // float seedSD_mdOut_ntp = 1;
//        // float seedSD_mdOut_itpLL = itpOL;
//        // if (itpOL >= 0 && itpOU >= 0)
//        // {
//        //     if (itpOL == itpOU)
//        //     {
//        //         seedSD_mdOut_ntp = 2;
//        //     }
//        // }
//        // else if (itpOL == -1 && itpOU == -1)
//        // {
//        //     seedSD_mdOut_ntp = 0;
//        // }
//        // else if (itpOU >= 0)
//        // {
//        //     seedSD_mdOut_itp = itpOU;
//        // }
//
//        //float seedSD_iRef = iSeed;
//        //float seedSD_iOut = iSeed;
//        TVector3 seedSD_r3 = r3LH;
//        //float seedSD_rt = r3LH.Pt();
//        //float seedSD_rtInv = 1.f / seedSD_rt;
//        //float seedSD_z = seedSD_r3.Z();
//        TVector3 seedSD_p3 = p3LH;
//        //float seedSD_alpha = r3LH.DeltaPhi(p3LH);
//        //float seedSD_dr = (r3LH - r3PCA).Pt();
//        //float seedSD_d = seedSD_rt - r3PCA.Pt();
//        //float seedSD_zeta = seedSD_p3.Pt() / seedSD_p3.Z();
//
//        // std::map<int, int> tps;
//        // int seedSD_itp = -1;
//        // int seedSD_ntp = 0;
//        // tps[itpRL]++;
//        // tps[itpRU]++;
//        // tps[itpOL]++;
//        // tps[itpOU]++;
//
//        // for (auto m : tps)
//        // {
//        //     if (m.first >= 0 && m.second > seedSD_ntp)
//        //     {
//        //         seedSD_itp = m.first;
//        //         seedSD_ntp = m.second;
//        //     }
//        // }
//
//        // // LL indexing is not very useful for seeds; to it anyways for "full" coverage
//        // int seedSD_itpLL = -1;
//        // int seedSD_ntpLL = 0;
//        // tps.clear();
//        // tps[itpRL]++;
//        // tps[itpOL]++;
//        // for (auto m : tps)
//        // {
//        //     if (m.first >= 0 && m.second > seedSD_ntpLL)
//        //     {
//        //         seedSD_itpLL = m.first;
//        //         seedSD_ntpLL = m.second;
//        //     }
//        // }
//
//        // For this segment in this layer, I need to add this many
//        // I suppose for now... I can have a SDL::Module of ginormous size
//        // with a special detId of ... 1? (does this require Module.h modification?)
//        // SDL::Hit
//        // SDL::Hit
//        // SDL::MiniDoublet
//        // SDL::Hit
//        // SDL::Hit
//        // SDL::MiniDoublet
//        // SDL::Segment
//        // std::cout <<  " seedSD_mdRef_rt: " << seedSD_mdRef_rt <<  std::endl;
//        // std::cout <<  " seedSD_mdRef_z: " << seedSD_mdRef_z <<  std::endl;
//        // std::cout <<  " seedSD_mdRef_alpha: " << seedSD_mdRef_alpha <<  std::endl;
//        // std::cout <<  " seedSD_mdOut_rt: " << seedSD_mdOut_rt <<  std::endl;
//        // std::cout <<  " seedSD_mdOut_z: " << seedSD_mdOut_z <<  std::endl;
//        // std::cout <<  " seedSD_mdOut_alpha: " << seedSD_mdOut_alpha <<  std::endl;
//        // std::cout <<  " trk.pix_x()[trk.see_hitIdx()[iSeed][0]]: " << trk.pix_x()[trk.see_hitIdx()[iSeed][0]] <<  std::endl;
//        // std::cout <<  " trk.pix_y()[trk.see_hitIdx()[iSeed][0]]: " << trk.pix_y()[trk.see_hitIdx()[iSeed][0]] <<  std::endl;
//        // std::cout <<  " sqrt(pow(trk.pix_x()[trk.see_hitIdx()[iSeed][0]],2)+pow(trk.pix_y()[trk.see_hitIdx()[iSeed][0]],2)): " << sqrt(pow(trk.pix_x()[trk.see_hitIdx()[iSeed][0]],2)+pow(trk.pix_y()[trk.see_hitIdx()[iSeed][0]],2)) <<  std::endl;
//        // std::cout <<  " trk.pix_z()[trk.see_hitIdx()[iSeed][0]]: " << trk.pix_z()[trk.see_hitIdx()[iSeed][0]] <<  std::endl;
//        // std::cout <<  " trk.pix_x()[trk.see_hitIdx()[iSeed][1]]: " << trk.pix_x()[trk.see_hitIdx()[iSeed][1]] <<  std::endl;
//        // std::cout <<  " trk.pix_y()[trk.see_hitIdx()[iSeed][1]]: " << trk.pix_y()[trk.see_hitIdx()[iSeed][1]] <<  std::endl;
//        // std::cout <<  " sqrt(pow(trk.pix_x()[trk.see_hitIdx()[iSeed][1]],2)+pow(trk.pix_y()[trk.see_hitIdx()[iSeed][1]],2)): " << sqrt(pow(trk.pix_x()[trk.see_hitIdx()[iSeed][1]],2)+pow(trk.pix_y()[trk.see_hitIdx()[iSeed][1]],2)) <<  std::endl;
//        // std::cout <<  " trk.pix_z()[trk.see_hitIdx()[iSeed][1]]: " << trk.pix_z()[trk.see_hitIdx()[iSeed][1]] <<  std::endl;
//        // std::cout <<  " trk.pix_x()[trk.see_hitIdx()[iSeed][2]]: " << trk.pix_x()[trk.see_hitIdx()[iSeed][2]] <<  std::endl;
//        // std::cout <<  " trk.pix_y()[trk.see_hitIdx()[iSeed][2]]: " << trk.pix_y()[trk.see_hitIdx()[iSeed][2]] <<  std::endl;
//        // std::cout <<  " sqrt(pow(trk.pix_x()[trk.see_hitIdx()[iSeed][2]],2)+pow(trk.pix_y()[trk.see_hitIdx()[iSeed][2]],2)): " << sqrt(pow(trk.pix_x()[trk.see_hitIdx()[iSeed][2]],2)+pow(trk.pix_y()[trk.see_hitIdx()[iSeed][2]],2)) <<  std::endl;
//        // std::cout <<  " trk.pix_z()[trk.see_hitIdx()[iSeed][2]]: " << trk.pix_z()[trk.see_hitIdx()[iSeed][2]] <<  std::endl;
//        // std::cout <<  " trk.pix_x()[trk.see_hitIdx()[iSeed][3]]: " << trk.pix_x()[trk.see_hitIdx()[iSeed][3]] <<  std::endl;
//        // std::cout <<  " trk.pix_y()[trk.see_hitIdx()[iSeed][3]]: " << trk.pix_y()[trk.see_hitIdx()[iSeed][3]] <<  std::endl;
//        // std::cout <<  " sqrt(pow(trk.pix_x()[trk.see_hitIdx()[iSeed][3]],2)+pow(trk.pix_y()[trk.see_hitIdx()[iSeed][3]],2)): " << sqrt(pow(trk.pix_x()[trk.see_hitIdx()[iSeed][3]],2)+pow(trk.pix_y()[trk.see_hitIdx()[iSeed][3]],2)) <<  std::endl;
//        // std::cout <<  " trk.pix_z()[trk.see_hitIdx()[iSeed][3]]: " << trk.pix_z()[trk.see_hitIdx()[iSeed][3]] <<  std::endl;
//
//        // Inner most hit
//        std::vector<SDL::CPU::Hit> hits;
//        int hitidx0 = trk.see_hitIdx()[iSeed][0];
//        //int hittype0 = trk.see_hitType()[iSeed][0];
//        // hits.push_back(SDL::CPU::Hit(trk.pix_x()[hitidx0], trk.pix_y()[hitidx0], trk.pix_z()[hitidx0], hitidx0));
//        // hits.push_back(SDL::CPU::Hit(r3PCA.X(), r3PCA.Y(), r3PCA.Z(), hitidx0));
//        hits.push_back(SDL::CPU::Hit(p3PCA.Pt(), p3PCA.Eta(), p3PCA.Phi(), hitidx0));
//        // hits.push_back(SDL::CPU::Hit(1, 2, 3, hitidx0));
//        int hitidx1 = trk.see_hitIdx()[iSeed][1];
//        //int hittype1 = trk.see_hitType()[iSeed][1];
//        // hits.push_back(SDL::CPU::Hit(trk.pix_x()[hitidx1], trk.pix_y()[hitidx1], trk.pix_z()[hitidx1], hitidx1));
//        hits.push_back(SDL::CPU::Hit(r3PCA.X(), r3PCA.Y(), r3PCA.Z(), hitidx1));
//        // hits.push_back(SDL::CPU::Hit(4, 5, 6, hitidx1));
//        int hitidx2 = trk.see_hitIdx()[iSeed][2];
//        //int hittype2 = trk.see_hitType()[iSeed][2];
//        // hits.push_back(SDL::CPU::Hit(trk.pix_x()[hitidx2], trk.pix_y()[hitidx2], trk.pix_z()[hitidx2], hitidx2));
//        hits.push_back(SDL::CPU::Hit(r3LH.X(), trk.see_dxy()[iSeed], trk.see_dz()[iSeed], hitidx2));
//        // hits.push_back(SDL::CPU::Hit(r3LH.X(), r3LH.Y(), r3LH.Z(), hitidx2));
//        // hits.push_back(SDL::CPU::Hit(7, 8, 9, hitidx2));
//        int hitidx3 = trk.see_hitIdx()[iSeed].size() > 3 ? trk.see_hitIdx()[iSeed][3] : trk.see_hitIdx()[iSeed][2]; // repeat last one if triplet
//        //int hittype3 = trk.see_hitIdx()[iSeed].size() > 3 ? trk.see_hitType()[iSeed][3] : trk.see_hitIdx()[iSeed][2]; // repeat last one if triplet
//        // hits.push_back(SDL::CPU::Hit(trk.pix_x()[hitidx3], trk.pix_y()[hitidx3], trk.pix_z()[hitidx3], hitidx3));
//        hits.push_back(SDL::CPU::Hit(r3LH.X(), r3LH.Y(), r3LH.Z(), hitidx3));
//        // hits.push_back(SDL::CPU::Hit(10, 11, 12, hitidx3));
//
//        float pixelSegmentDeltaPhiChange = r3LH.DeltaPhi(p3LH);
//        float ptIn = p3LH.Pt();
//        float ptErr = trk.see_ptErr()[iSeed];
//        float etaErr = trk.see_etaErr()[iSeed];
//        float px = p3LH.X();
//        float py = p3LH.Y();
//        float pz = p3LH.Z();
//
//        if ((ptIn > 1 - 2 * ptErr) and (fabs(p3LH.Eta()) < 3))
//            event.addPixelSegmentsToEvent(hits, pixelSegmentDeltaPhiChange, ptIn, ptErr, px, py, pz, etaErr, iSeed);
//
//    }
//
//    float pls_loading_elapsed = my_timer.RealTime();
//    if (ana.verbose >= 2) std::cout << "Loading pixel line segments processing time: " << pls_loading_elapsed << " secs" << std::endl;
//    return pls_loading_elapsed;
//}
//
////__________________________________________________________________________________________
//float runMiniDoublet_on_CPU(SDL::CPU::Event& event)
//{
//    TStopwatch my_timer;
//    if (ana.verbose >= 2) std::cout << "Reco Mini-Doublet start" << std::endl;
//    my_timer.Start();
//    event.createMiniDoublets();
//    float md_elapsed = my_timer.RealTime();
//    if (ana.verbose >= 2) std::cout << "Reco Mini-doublet processing time: " << md_elapsed << " secs" << std::endl;
//    return md_elapsed;
//}
//
////__________________________________________________________________________________________
//float runSegment_on_CPU(SDL::CPU::Event& event)
//{
//    TStopwatch my_timer;
//    if (ana.verbose >= 2) std::cout << "Reco Segment start" << std::endl;
//    my_timer.Start();
//    event.createSegmentsWithModuleMap();
//    float sg_elapsed = my_timer.RealTime();
//    if (ana.verbose >= 2) std::cout << "Reco Segment processing time: " << sg_elapsed << " secs" << std::endl;
//    return sg_elapsed;
//}
//
////__________________________________________________________________________________________
//float runT4_on_CPU(SDL::CPU::Event& event)
//{
//    TStopwatch my_timer;
//    if (ana.verbose >= 2) std::cout << "Reco Quadruplet T4 start" << std::endl;
//    my_timer.Start();
//    event.createTrackletsWithModuleMap();
//    float t4_elapsed = my_timer.RealTime();
//    if (ana.verbose >= 2) std::cout << "Reco T4 (just T4) processing time: " << t4_elapsed << " secs" << std::endl;
//    return t4_elapsed;
//}
//
////__________________________________________________________________________________________
//float runT4x_on_CPU(SDL::CPU::Event& event)
//{
//    TStopwatch my_timer;
//    if (ana.verbose >= 2) std::cout << "Reco Quadruplet T4x start" << std::endl;
//    my_timer.Start();
//    event.createTrackletsWithAGapWithModuleMap();
//    float t4x_elapsed = my_timer.RealTime();
//    if (ana.verbose >= 2) std::cout << "Reco T4x (just T4x) processing time: " << t4x_elapsed << " secs" << std::endl;
//    return t4x_elapsed;
//}
//
////__________________________________________________________________________________________
//float runpT4_on_CPU(SDL::CPU::Event& event)
//{
//    TStopwatch my_timer;
//    if (ana.verbose >= 2) std::cout << "Reco Quadruplet pT4 start" << std::endl;
//    my_timer.Start();
//    // event.createTrackletsWithPixelLineSegments();
//    event.createTrackletsWithPixelLineSegments_v2();
//    float pt4_elapsed = my_timer.RealTime();
//    if (ana.verbose >= 2) std::cout << "Reco pT4 processing time: " << pt4_elapsed << " secs" << std::endl;
//    return pt4_elapsed;
//}
//
////__________________________________________________________________________________________
//float runT3_on_CPU(SDL::CPU::Event& event)
//{
//    TStopwatch my_timer;
//    if (ana.verbose >= 2) std::cout << "Reco T3 start" << std::endl;
//    my_timer.Start();
//    event.createTriplets();
//    float t3_elapsed = my_timer.RealTime();
//    if (ana.verbose >= 2) std::cout << "Reco T3 processing time: " << t3_elapsed<< " secs" << std::endl;
//    return t3_elapsed;
//}
//
////__________________________________________________________________________________________
//float runTrackCandidate_on_CPU(SDL::CPU::Event& event)
//{
//    TStopwatch my_timer;
//    if (ana.verbose >= 2) std::cout << "Reco TrackCandidate start" << std::endl;
//    my_timer.Start();
//    event.createTrackCandidates();
//    float tc_elapsed = my_timer.RealTime();
//    if (ana.verbose >= 2) std::cout << "Reco TrackCandidate processing time: " << tc_elapsed << " secs" << std::endl;
//    return tc_elapsed;
//}
//
////__________________________________________________________________________________________
//float runT5_on_CPU(SDL::CPU::Event& event)
//{
//    TStopwatch my_timer;
//    if (ana.verbose >= 2) std::cout << "Reco T5 start" << std::endl;
//    my_timer.Start();
//    event.createT5s();
//    float t5_elapsed = my_timer.RealTime();
//    if (ana.verbose >= 2) std::cout << "Reco T5 processing time: " << t5_elapsed << " secs" << std::endl;
//    return t5_elapsed;
//}
//
////__________________________________________________________________________________________
//float runpT3_on_CPU(SDL::CPU::Event& event)
//{
//    TStopwatch my_timer;
//    if (ana.verbose >= 2) std::cout << "Reco pT3 start" << std::endl;
//    my_timer.Start();
//    event.createpT3s();
//    float pt3_elapsed = my_timer.RealTime();
//    if (ana.verbose >= 2) std::cout << "Reco pT3 processing time: " << pt3_elapsed << " secs" << std::endl;
//    return pt3_elapsed;
//}
//
////__________________________________________________________________________________________
//void printHitSummary(SDL::CPU::Event& event)
//{
//    if (ana.verbose >= 2) std::cout << "Summary of hits" << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Hits: " << event.getNumberOfHits() << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Hits in barrel layer 1: " << event.getNumberOfHitsByLayerBarrel(0) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Hits in barrel layer 2: " << event.getNumberOfHitsByLayerBarrel(1) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Hits in barrel layer 3: " << event.getNumberOfHitsByLayerBarrel(2) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Hits in barrel layer 4: " << event.getNumberOfHitsByLayerBarrel(3) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Hits in barrel layer 5: " << event.getNumberOfHitsByLayerBarrel(4) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Hits in barrel layer 6: " << event.getNumberOfHitsByLayerBarrel(5) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Hits in endcap layer 1: " << event.getNumberOfHitsByLayerEndcap(0) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Hits in endcap layer 2: " << event.getNumberOfHitsByLayerEndcap(1) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Hits in endcap layer 3: " << event.getNumberOfHitsByLayerEndcap(2) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Hits in endcap layer 4: " << event.getNumberOfHitsByLayerEndcap(3) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Hits in endcap layer 5: " << event.getNumberOfHitsByLayerEndcap(4) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Hits Upper Module in layer 1: " << event.getNumberOfHitsByLayerBarrelUpperModule(0) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Hits Upper Module in layer 2: " << event.getNumberOfHitsByLayerBarrelUpperModule(1) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Hits Upper Module in layer 3: " << event.getNumberOfHitsByLayerBarrelUpperModule(2) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Hits Upper Module in layer 4: " << event.getNumberOfHitsByLayerBarrelUpperModule(3) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Hits Upper Module in layer 5: " << event.getNumberOfHitsByLayerBarrelUpperModule(4) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Hits Upper Module in layer 6: " << event.getNumberOfHitsByLayerBarrelUpperModule(5) << std::endl;
//}
//
////__________________________________________________________________________________________
//void printMiniDoubletSummary(SDL::CPU::Event& event)
//{
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced: " << event.getNumberOfMiniDoublets() << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced barrel layer 1: " << event.getNumberOfMiniDoubletsByLayerBarrel(0) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced barrel layer 2: " << event.getNumberOfMiniDoubletsByLayerBarrel(1) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced barrel layer 3: " << event.getNumberOfMiniDoubletsByLayerBarrel(2) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced barrel layer 4: " << event.getNumberOfMiniDoubletsByLayerBarrel(3) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced barrel layer 5: " << event.getNumberOfMiniDoubletsByLayerBarrel(4) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced barrel layer 6: " << event.getNumberOfMiniDoubletsByLayerBarrel(5) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced endcap layer 1: " << event.getNumberOfMiniDoubletsByLayerEndcap(0) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced endcap layer 2: " << event.getNumberOfMiniDoubletsByLayerEndcap(1) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced endcap layer 3: " << event.getNumberOfMiniDoubletsByLayerEndcap(2) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced endcap layer 4: " << event.getNumberOfMiniDoubletsByLayerEndcap(3) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets produced endcap layer 5: " << event.getNumberOfMiniDoubletsByLayerEndcap(4) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets considered: " << event.getNumberOfMiniDoubletCandidates() << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets considered barrel layer 1: " << event.getNumberOfMiniDoubletCandidatesByLayerBarrel(0) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets considered barrel layer 2: " << event.getNumberOfMiniDoubletCandidatesByLayerBarrel(1) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets considered barrel layer 3: " << event.getNumberOfMiniDoubletCandidatesByLayerBarrel(2) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets considered barrel layer 4: " << event.getNumberOfMiniDoubletCandidatesByLayerBarrel(3) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets considered barrel layer 5: " << event.getNumberOfMiniDoubletCandidatesByLayerBarrel(4) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets considered barrel layer 6: " << event.getNumberOfMiniDoubletCandidatesByLayerBarrel(5) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets considered endcap layer 1: " << event.getNumberOfMiniDoubletCandidatesByLayerEndcap(0) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets considered endcap layer 2: " << event.getNumberOfMiniDoubletCandidatesByLayerEndcap(1) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets considered endcap layer 3: " << event.getNumberOfMiniDoubletCandidatesByLayerEndcap(2) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets considered endcap layer 4: " << event.getNumberOfMiniDoubletCandidatesByLayerEndcap(3) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Mini-doublets considered endcap layer 5: " << event.getNumberOfMiniDoubletCandidatesByLayerEndcap(4) << std::endl;
//}
//
////__________________________________________________________________________________________
//void printSegmentSummary(SDL::CPU::Event& event)
//{
//    if (ana.verbose >= 2) std::cout << "# of Pixel Segments: " << event.getPixelLayer().getSegmentPtrs().size() << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Segments produced: " << event.getNumberOfSegments() << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Segments produced barrel layer 1-2: " << event.getNumberOfSegmentsByLayerBarrel(0) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Segments produced barrel layer 2-3: " << event.getNumberOfSegmentsByLayerBarrel(1) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Segments produced barrel layer 3-4: " << event.getNumberOfSegmentsByLayerBarrel(2) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Segments produced barrel layer 4-5: " << event.getNumberOfSegmentsByLayerBarrel(3) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Segments produced barrel layer 5-6: " << event.getNumberOfSegmentsByLayerBarrel(4) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Segments produced endcap layer 1-2: " << event.getNumberOfSegmentsByLayerEndcap(0) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Segments produced endcap layer 2-3: " << event.getNumberOfSegmentsByLayerEndcap(1) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Segments produced endcap layer 3-4: " << event.getNumberOfSegmentsByLayerEndcap(2) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Segments produced endcap layer 4-5: " << event.getNumberOfSegmentsByLayerEndcap(3) << std::endl;
//    // if (ana.verbose 2= 1) std::cout << "# of Segments produced layer 6: " << event.getNumberOfSegmentsByLayerBarrel(5) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Segments considered: " << event.getNumberOfSegmentCandidates() << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Segments considered layer 1-2: " << event.getNumberOfSegmentCandidatesByLayerBarrel(0) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Segments considered layer 2-3: " << event.getNumberOfSegmentCandidatesByLayerBarrel(1) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Segments considered layer 3-4: " << event.getNumberOfSegmentCandidatesByLayerBarrel(2) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Segments considered layer 4-5: " << event.getNumberOfSegmentCandidatesByLayerBarrel(3) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Segments considered layer 5-6: " << event.getNumberOfSegmentCandidatesByLayerBarrel(4) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Segments considered layer 1-2: " << event.getNumberOfSegmentCandidatesByLayerEndcap(0) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Segments considered layer 2-3: " << event.getNumberOfSegmentCandidatesByLayerEndcap(1) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Segments considered layer 3-4: " << event.getNumberOfSegmentCandidatesByLayerEndcap(2) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Segments considered layer 4-5: " << event.getNumberOfSegmentCandidatesByLayerEndcap(3) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of Segments considered layer 6: " << event.getNumberOfSegmentCandidatesByLayerBarrel(5) << std::endl;
//}
//
////__________________________________________________________________________________________
//void printTripletSummary(SDL::CPU::Event& event)
//{
//    // ----------------
//    if (ana.verbose >= 2) std::cout << "# of Triplets produced: " << event.getNumberOfTriplets() << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Triplets produced layer 1-2-3: " << event.getNumberOfTripletsByLayerBarrel(0) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Triplets produced layer 2-3-4: " << event.getNumberOfTripletsByLayerBarrel(1) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Triplets produced layer 3-4-5: " << event.getNumberOfTripletsByLayerBarrel(2) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Triplets produced layer 4-5-6: " << event.getNumberOfTripletsByLayerBarrel(3) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of Triplets produced layer 5: " << event.getNumberOfTripletsByLayerBarrel(4) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of Triplets produced layer 6: " << event.getNumberOfTripletsByLayerBarrel(5) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Triplets considered: " << event.getNumberOfTripletCandidates() << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Triplets considered layer 1-2-3: " << event.getNumberOfTripletCandidatesByLayerBarrel(0) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Triplets considered layer 2-3-4: " << event.getNumberOfTripletCandidatesByLayerBarrel(1) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Triplets considered layer 3-4-5: " << event.getNumberOfTripletCandidatesByLayerBarrel(2) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Triplets considered layer 4-5-6: " << event.getNumberOfTripletCandidatesByLayerBarrel(3) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of Triplets considered layer 5: " << event.getNumberOfTripletCandidatesByLayerBarrel(4) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of Triplets considered layer 6: " << event.getNumberOfTripletCandidatesByLayerBarrel(5) << std::endl;
//    // ----------------
//}
//
////__________________________________________________________________________________________
//void printTrackletSummary(SDL::CPU::Event& event)
//{
//    // ----------------
//    if (ana.verbose >= 2) std::cout << "# of Tracklets produced with pixel segment: " << event.getPixelLayer().getTrackletPtrs().size() << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Tracklets produced: " << event.getNumberOfTracklets() << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Tracklets produced layer 1-2-3-4: " << event.getNumberOfTrackletsByLayerBarrel(0) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Tracklets produced layer 2-3-4-5: " << event.getNumberOfTrackletsByLayerBarrel(1) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Tracklets produced layer 3-4-5-6: " << event.getNumberOfTrackletsByLayerBarrel(2) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Tracklets produced layer 1-2-3-4: " << event.getNumberOfTrackletsByLayerEndcap(0) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Tracklets produced layer 2-3-4-5: " << event.getNumberOfTrackletsByLayerEndcap(1) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of Tracklets produced layer 4: " << event.getNumberOfTrackletsByLayerBarrel(3) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of Tracklets produced layer 5: " << event.getNumberOfTrackletsByLayerBarrel(4) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of Tracklets produced layer 6: " << event.getNumberOfTrackletsByLayerBarrel(5) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Tracklets considered: " << event.getNumberOfTrackletCandidates() << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Tracklets considered layer 1-2-3-4: " << event.getNumberOfTrackletCandidatesByLayerBarrel(0) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Tracklets considered layer 2-3-4-5: " << event.getNumberOfTrackletCandidatesByLayerBarrel(1) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Tracklets considered layer 3-4-5-6: " << event.getNumberOfTrackletCandidatesByLayerBarrel(2) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Tracklets considered layer 1-2-3-4: " << event.getNumberOfTrackletCandidatesByLayerEndcap(0) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of Tracklets considered layer 2-3-4-5: " << event.getNumberOfTrackletCandidatesByLayerEndcap(1) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of Tracklets considered layer 4: " << event.getNumberOfTrackletCandidatesByLayerBarrel(3) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of Tracklets considered layer 5: " << event.getNumberOfTrackletCandidatesByLayerBarrel(4) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of Tracklets considered layer 6: " << event.getNumberOfTrackletCandidatesByLayerBarrel(5) << std::endl;
//    // ----------------
//}
//
////__________________________________________________________________________________________
//void printTrackCandidateSummary(SDL::CPU::Event& event)
//{
//    // ----------------
//    if (ana.verbose >= 2) std::cout << "# of TrackCandidates produced: " << event.getNumberOfTrackCandidates() << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of TrackCandidates produced layer 1-2-3-4-5-6: " << event.getNumberOfTrackCandidatesByLayerBarrel(0) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of TrackCandidates produced layer 2: " << event.getNumberOfTrackCandidatesByLayerBarrel(1) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of TrackCandidates produced layer 3: " << event.getNumberOfTrackCandidatesByLayerBarrel(2) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of TrackCandidates produced layer 4: " << event.getNumberOfTrackCandidatesByLayerBarrel(3) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of TrackCandidates produced layer 5: " << event.getNumberOfTrackCandidatesByLayerBarrel(4) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of TrackCandidates produced layer 6: " << event.getNumberOfTrackCandidatesByLayerBarrel(5) << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of TrackCandidates considered: " << event.getNumberOfTrackCandidateCandidates() << std::endl;
//    if (ana.verbose >= 2) std::cout << "# of TrackCandidates considered layer 1-2-3-4-5-6: " << event.getNumberOfTrackCandidateCandidatesByLayerBarrel(0) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of TrackCandidates considered layer 2: " << event.getNumberOfTrackCandidateCandidatesByLayerBarrel(1) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of TrackCandidates considered layer 3: " << event.getNumberOfTrackCandidateCandidatesByLayerBarrel(2) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of TrackCandidates considered layer 4: " << event.getNumberOfTrackCandidateCandidatesByLayerBarrel(3) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of TrackCandidates considered layer 5: " << event.getNumberOfTrackCandidateCandidatesByLayerBarrel(4) << std::endl;
//    // if (ana.verbose >= 1) std::cout << "# of TrackCandidates considered layer 6: " << event.getNumberOfTrackCandidateCandidatesByLayerBarrel(5) << std::endl;
//    // ----------------
//
//}

//__________________________________________________________________________________________
bool isDenomSimTrk(int isimtrk)
{
    if (isimtrk < 0)
        return false;
    //const float& pt = trk.sim_pt()[isimtrk];
    //const float& eta = trk.sim_eta()[isimtrk];
    //const float& dz = trk.sim_pca_dz()[isimtrk];
    //const float& dxy = trk.sim_pca_dxy()[isimtrk];
    //const float& phi = trk.sim_phi()[isimtrk];
    const int& bunch = trk.sim_bunchCrossing()[isimtrk];
    const int& event = trk.sim_event()[isimtrk];
    //const int& vtxIdx = trk.sim_parentVtxIdx()[isimtrk];
    //const int& pdgidtrk = trk.sim_pdgId()[isimtrk];
    const int& q = trk.sim_q()[isimtrk];
    //const float& vtx_x = trk.simvtx_x()[vtxIdx];
    //const float& vtx_y = trk.simvtx_y()[vtxIdx];
    //const float& vtx_z = trk.simvtx_z()[vtxIdx];
    //const float& vtx_perp = sqrt(vtx_x * vtx_x + vtx_y * vtx_y);

    if (bunch != 0)
        return false;

    if (event != 0)
        return false;

    if (q == 0)
        return false;

    return true;
}

//__________________________________________________________________________________________
bool isDenomOfInterestSimTrk(int isimtrk)
{
    if (isimtrk < 0)
        return false;
    const float& pt = trk.sim_pt()[isimtrk];
    if (pt < 1)
        return false;
    //const float& eta = trk.sim_eta()[isimtrk];
    //const float& dz = trk.sim_pca_dz()[isimtrk];
    //const float& dxy = trk.sim_pca_dxy()[isimtrk];
    //const float& phi = trk.sim_phi()[isimtrk];
    const int& bunch = trk.sim_bunchCrossing()[isimtrk];
    //const int& event = trk.sim_event()[isimtrk];
    const int& vtxIdx = trk.sim_parentVtxIdx()[isimtrk];
    //const int& pdgidtrk = trk.sim_pdgId()[isimtrk];
    const int& q = trk.sim_q()[isimtrk];
    const float& vtx_x = trk.simvtx_x()[vtxIdx];
    const float& vtx_y = trk.simvtx_y()[vtxIdx];
    const float& vtx_z = trk.simvtx_z()[vtxIdx];
    const float& vtx_perp = sqrt(vtx_x * vtx_x + vtx_y * vtx_y);

    if (bunch != 0)
        return false;

    if (q == 0)
        return false;

    if (vtx_perp > 2.5)
        return false;

    if (abs(vtx_z) > 30)
        return false;

    return true;
}


//__________________________________________________________________________________________
int getDenomSimTrkType(int isimtrk)
{
    if (isimtrk < 0)
        return 0; // not a sim
    const int& q = trk.sim_q()[isimtrk];
    if (q == 0)
        return 1; // sim
    const float& pt = trk.sim_pt()[isimtrk];
    const float& eta = trk.sim_eta()[isimtrk];
    if (pt < 1 or abs(eta) > 2.4)
        return 2; // sim and charged
    //const float& dz = trk.sim_pca_dz()[isimtrk];
    //const float& dxy = trk.sim_pca_dxy()[isimtrk];
    //const float& phi = trk.sim_phi()[isimtrk];
    const int& bunch = trk.sim_bunchCrossing()[isimtrk];
    const int& event = trk.sim_event()[isimtrk];
    const int& vtxIdx = trk.sim_parentVtxIdx()[isimtrk];
    //const int& pdgidtrk = trk.sim_pdgId()[isimtrk];
    const float& vtx_x = trk.simvtx_x()[vtxIdx];
    const float& vtx_y = trk.simvtx_y()[vtxIdx];
    const float& vtx_z = trk.simvtx_z()[vtxIdx];
    const float& vtx_perp = sqrt(vtx_x * vtx_x + vtx_y * vtx_y);

    if (vtx_perp > 2.5)
        return 3; // pt > 1 and abs(eta) < 2.4

    if (abs(vtx_z) > 30)
        return 4; // pt > 1 and abs(eta) < 2.4 and vtx < 2.5

    if (bunch != 0)
        return 5; // pt > 1 and abs(eta) < 2.4 and vtx < 2.5 and vtx < 300

    if (event != 0)
        return 6; // pt > 1 and abs(eta) < 2.4 and vtx 2.5/30 and bunch == 0

    return 7; // pt > 1 and abs(eta) < 2.4 and vtx 2.5/30 and bunch == 0 and event == 0
}

//__________________________________________________________________________________________
int bestSimHitMatch(int irecohit)
{
    const std::vector<int>& simhitidxs = trk.ph2_simHitIdx()[irecohit];
    if (simhitidxs.size() == 0)
        return -999;
    float ptmax = 0;
    int best_i = -1;
    for (auto& simhitidx : simhitidxs)
    {
        int isimtrk = trk.simhit_simTrkIdx()[simhitidx];
        float tmppt = trk.sim_pt()[isimtrk];
        if (tmppt > ptmax)
        {
            best_i = simhitidx;
            ptmax = tmppt;
        }
    }
    return best_i;
}

//__________________________________________________________________________________________
//int logicalLayer(const SDL::CPU::Module& module)
//{
//    return module.layer() + 6 * (module.subdet() == 4) + 5 * (module.subdet() == 4 and module.moduleType() == 1);
//}
//
////__________________________________________________________________________________________
//int isAnchorLayer(const SDL::CPU::Module& module)
//{
//
//    if (module.moduleType() == SDL::CPU::Module::PS)
//    {
//        if (module.moduleLayerType() == SDL::CPU::Module::Pixel)
//        {
//            return true;
//        }
//        else
//        {
//            return false;
//        }
//    }
//    else
//    {
//        return module.isLower();
//    }
//}

//__________________________________________________________________________________________
TString get_absolute_path_after_check_file_exists(const std::string name)
{
    std::filesystem::path fullpath = std::filesystem::absolute(name.c_str());
    // std::cout << "Checking file path = " << fullpath << std::endl;
    // std::cout <<  " fullpath.string().c_str(): " << fullpath.string().c_str() <<  std::endl;
    if (not std::filesystem::exists(fullpath))
    {
        std::cout << "ERROR: Could not find the file = " << fullpath << std::endl;
        exit(2);
    }
    return TString(fullpath.string().c_str());
}
