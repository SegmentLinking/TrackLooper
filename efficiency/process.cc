#include "process.h"
#include "../src/SDLMath.h"

#include "sdl_types.h"

// ./process INPUTFILEPATH OUTPUTFILE [NEVENTS]
int main(int argc, char** argv)
{

    // Parse arguments
    parseArguments(argc, argv);

    // Initialize input and output root files
    initializeInputsAndOutputs();

    // Create various bits important for each track
    createSDLVariables();

    // creating a set of efficiency plots
    std::vector<EfficiencySetDefinition> list_effSetDef;

    // Book MD
    for (auto& md_type : MD_types)
        list_effSetDef.push_back(EfficiencySetDefinition(TString::Format("MD_%s", md_type.Data()), 13, [&, md_type](int isim) {return ana.tx.getBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_MD_%s", md_type.Data()))[isim].size() > 0;}));

    for (auto& ls_type : LS_types)
        list_effSetDef.push_back(EfficiencySetDefinition(TString::Format("LS_%s", ls_type.Data()), 13, [&, ls_type](int isim) {return ana.tx.getBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_LS_%s", ls_type.Data()))[isim].size() > 0;}));

    list_effSetDef.push_back(EfficiencySetDefinition("pLS_P", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_pLS_P")[isim].size() > 0;}));

    for (auto& pt4_type : pT4_types)
        list_effSetDef.push_back(EfficiencySetDefinition(TString::Format("pT4_%s", pt4_type.Data()), 13, [&, pt4_type](int isim) {return ana.tx.getBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_pT4_%s", pt4_type.Data()))[isim].size() > 0;}));
    list_effSetDef.push_back(EfficiencySetDefinition("pT4_AllTypes", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_pT4_AllTypes")[isim].size() > 0;}));

    for (auto& t4_type : T4_types)
        list_effSetDef.push_back(EfficiencySetDefinition(TString::Format("T4_%s", t4_type.Data()), 13, [&, t4_type](int isim) {return ana.tx.getBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_T4_%s", t4_type.Data()))[isim].size() > 0;}));
    list_effSetDef.push_back(EfficiencySetDefinition("T4_AllTypes", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_T4_AllTypes")[isim].size() > 0;}));

    for (auto& t4x_type : T4x_types)
        list_effSetDef.push_back(EfficiencySetDefinition(TString::Format("T4x_%s", t4x_type.Data()), 13, [&, t4x_type](int isim) {return ana.tx.getBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_T4x_%s", t4x_type.Data()))[isim].size() > 0;}));
    list_effSetDef.push_back(EfficiencySetDefinition("T4x_AllTypes", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_T4x_AllTypes")[isim].size() > 0;}));
    list_effSetDef.push_back(EfficiencySetDefinition("AllT4s_AllTypes", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_AllT4s_AllTypes")[isim].size() > 0;}));

    for (auto& t3_type : T3_types)
        list_effSetDef.push_back(EfficiencySetDefinition(TString::Format("T3_%s", t3_type.Data()), 13, [&, t3_type](int isim) {return ana.tx.getBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_T3_%s", t3_type.Data()))[isim].size() > 0;}));
    list_effSetDef.push_back(EfficiencySetDefinition("T3_AllTypes", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_T3_AllTypes")[isim].size() > 0;}));

    for (auto& tc_type : TC_types)
        list_effSetDef.push_back(EfficiencySetDefinition(TString::Format("TC_%s", tc_type.Data()), 13, [&, tc_type](int isim) {return ana.tx.getBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_TC_%s", tc_type.Data()))[isim].size() > 0;}));
    list_effSetDef.push_back(EfficiencySetDefinition("TC_AllTypes", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_TC_AllTypes")[isim].size() > 0;}));
    list_effSetDef.push_back(EfficiencySetDefinition("TC_Set1Types", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_TC_Set1Types")[isim].size() > 0;}));
    list_effSetDef.push_back(EfficiencySetDefinition("TC_Set2Types", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_TC_Set2Types")[isim].size() > 0;}));
    list_effSetDef.push_back(EfficiencySetDefinition("TC_Set3Types", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_TC_Set3Types")[isim].size() > 0;}));
    list_effSetDef.push_back(EfficiencySetDefinition("TC_Set4Types", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_TC_Set4Types")[isim].size() > 0;}));

    list_effSetDef.push_back(EfficiencySetDefinition("pix_P", 13, [&](int isim) {return ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_pix_P")[isim].size() > 0;}));

    bookEfficiencySets(list_effSetDef);

    ana.cutflow.bookCutflows();

    // Book Histograms
    ana.cutflow.bookHistograms(ana.histograms); // if just want to book everywhere

    // Looping input file
    while (ana.looper.nextEvent())
    {

        // If splitting jobs are requested then determine whether to process the event or not based on remainder
        if (ana.job_index != -1 and ana.nsplit_jobs != -1)
        {
            if (ana.looper.getNEventsProcessed() % ana.nsplit_jobs != (unsigned int) ana.job_index)
                continue;
        }

        // Reset all variables
        ana.tx.clear();

        setSDLVariables();

        fillEfficiencySets(list_effSetDef);

        //Do what you need to do in for each event here
        //To save use the following function
        ana.cutflow.fill();
    }

    // Writing output file
    ana.cutflow.saveOutput();

    // The below can be sometimes crucial
    delete ana.output_tfile;
}

void createSDLVariables()
{

    // hasAll12Hits
    ana.tx.createBranch<vector<int>>("sim_hasAll12HitsInBarrel");

    // Mini-Doublets
    for (auto& md_type : MD_types)
        ana.tx.createBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_MD_%s", md_type.Data()));

    // Line segments
    for (auto& ls_type : LS_types)
        ana.tx.createBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_LS_%s", ls_type.Data()));

    // Pixel Line segments
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_pLS_P");

    // Tracklets with Pixel
    for (auto& pt4_type : pT4_types)
        ana.tx.createBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_pT4_%s", pt4_type.Data()));
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_pT4_AllTypes");

    // Tracklets
    for (auto& t4_type : T4_types)
        ana.tx.createBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_T4_%s", t4_type.Data()));
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_T4_AllTypes");

    // Tracklets with a gap
    for (auto& t4x_type : T4x_types)
        ana.tx.createBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_T4x_%s", t4x_type.Data()));
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_T4x_AllTypes");
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_AllT4s_AllTypes");

    // Triplets
    for (auto& t3_type : T3_types)
        ana.tx.createBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_T3_%s", t3_type.Data()));
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_T3_AllTypes");

    // Track Candidates
    for (auto& tc_type : TC_types)
        ana.tx.createBranch<vector<vector<int>>>(TString::Format("mtv_match_idxs_TC_%s", tc_type.Data()));
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_TC_AllTypes");
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_TC_Set1Types");
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_TC_Set2Types");
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_TC_Set3Types");
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_TC_Set4Types");

    // Pixel Line segments
    ana.tx.createBranch<vector<vector<int>>>("mtv_match_idxs_pix_P");

}

void setSDLVariables()
{


    // First compute which sim trk idxs are matched to pixel seeds of the choice
    std::vector<int> unique_simtrkIdx_matched_to_seeds;
    for (unsigned int iseed = 0; iseed < sdl.see_hitIdx().size(); ++iseed)
    {

        bool good_seed_type = false;
        if (sdl.see_algo()[iseed] == 4) good_seed_type = true;
        if (sdl.see_algo()[iseed] == 5) good_seed_type = true;
        if (sdl.see_algo()[iseed] == 7) good_seed_type = true;
        if (sdl.see_algo()[iseed] == 22) good_seed_type = true;
        if (sdl.see_algo()[iseed] == 23) good_seed_type = true;
        if (sdl.see_algo()[iseed] == 24) good_seed_type = true;
        if (not good_seed_type)
            continue;

        for (auto& isim : sdl.see_simTrkIdx()[iseed])
        {
            if (std::find(unique_simtrkIdx_matched_to_seeds.begin(), unique_simtrkIdx_matched_to_seeds.end(), isim) == unique_simtrkIdx_matched_to_seeds.end())
            {
                unique_simtrkIdx_matched_to_seeds.push_back(isim);
            }
        }
    }

    for (unsigned int isim = 0; isim < sdl.sim_pt().size(); ++isim)
    {

        // hasAll12Hits
        ana.tx.pushbackToBranch<int>("sim_hasAll12HitsInBarrel", sdl.sim_hasAll12HitsInBarrel()[isim]);

        // Mini-Doublets
        std::array<std::vector<int>, n_MD_types> MD_idxs;
        for (auto& mdIdx : sdl.sim_mdIdx()[isim])
        {
            const int& layer = sdl.md_layer()[mdIdx][0];
            MD_idxs[layer-1].push_back(mdIdx);
        }

        // Set the MD idxs variables
        for (unsigned int imd = 0; imd < MD_types.size(); ++imd)
        {
            ana.tx.pushbackToBranch<vector<int>>(TString::Format("mtv_match_idxs_MD_%s", MD_types.at(imd).Data()), MD_idxs.at(imd));
        }

        // Line Segments
        std::array<std::vector<int>, n_LS_types> LS_idxs;
        for (auto& sgIdx : sdl.sim_sgIdx()[isim])
        {
            const int& layerIn = sdl.sg_layer()[sgIdx][0];
            const int& layerOut = sdl.sg_layer()[sgIdx][2];
            LS_idxs[LS_types_map[std::make_pair(layerIn, layerOut)]].push_back(sgIdx);
        }

        // Set the LS idxs variables
        for (unsigned int ils = 0; ils < LS_types.size(); ++ils)
        {
            ana.tx.pushbackToBranch<vector<int>>(TString::Format("mtv_match_idxs_LS_%s", LS_types.at(ils).Data()), LS_idxs.at(ils));
        }

        // Pixel Line segments
        std::vector<int> pLS_idxs;
        for (auto& psgIdx : sdl.sim_psgIdx()[isim])
        {
            pLS_idxs.push_back(psgIdx);
        }

        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_pLS_P", pLS_idxs);

        // Pixel (pT4)
        std::array<std::vector<int>, n_pT4_types> pT4_idxs;
        std::vector<int> pT4_idxs_all;
        for (auto& pqpIdx : sdl.sim_pqpIdx()[isim])
        {
            const int& layerOutLo = sdl.pqp_layer()[pqpIdx][2];
            const int& layerOutUp = sdl.pqp_layer()[pqpIdx][3];
            pT4_idxs[pT4_types_map[std::make_pair(layerOutLo, layerOutUp)]].push_back(pqpIdx);
            pT4_idxs_all.push_back(pqpIdx);
        }

        // Set the pT4 idxs variables
        for (unsigned int ipt4 = 0; ipt4 < pT4_types.size(); ++ipt4)
        {
            ana.tx.pushbackToBranch<vector<int>>(TString::Format("mtv_match_idxs_pT4_%s", pT4_types.at(ipt4).Data()), pT4_idxs.at(ipt4));
        }
        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_pT4_AllTypes", pT4_idxs_all);

        // Any T4 (not including pT4)
        std::vector<int> AllT4s_idxs_all;

        // Tracklet (T4)
        std::array<std::vector<int>, n_T4_types> T4_idxs;
        std::vector<int> T4_idxs_all;
        for (auto& qpIdx : sdl.sim_qpIdx()[isim])
        {
            std::vector<int> layers = {sdl.qp_layer()[qpIdx][0], sdl.qp_layer()[qpIdx][1], sdl.qp_layer()[qpIdx][2], sdl.qp_layer()[qpIdx][3]};
            if (T4_types_map.find(layers) != T4_types_map.end())
            {
                T4_idxs[T4_types_map.at(layers)].push_back(qpIdx);
                T4_idxs_all.push_back(qpIdx);
                AllT4s_idxs_all.push_back(qpIdx);
            }
        }

        // Set the T4 idxs variables
        for (unsigned int it4 = 0; it4 < T4_types.size(); ++it4)
        {
            ana.tx.pushbackToBranch<vector<int>>(TString::Format("mtv_match_idxs_T4_%s", T4_types.at(it4).Data()), T4_idxs.at(it4));
        }
        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_T4_AllTypes", T4_idxs_all);

        // Tracklet (T4x)
        std::array<std::vector<int>, n_T4x_types> T4x_idxs;
        std::vector<int> T4x_idxs_all;
        for (auto& qpIdx : sdl.sim_qpIdx()[isim])
        {
            std::vector<int> layers = {sdl.qp_layer()[qpIdx][0], sdl.qp_layer()[qpIdx][1], sdl.qp_layer()[qpIdx][2], sdl.qp_layer()[qpIdx][3]};
            if (T4x_types_map.find(layers) != T4x_types_map.end())
            {
                T4x_idxs[T4x_types_map.at(layers)].push_back(qpIdx);
                T4x_idxs_all.push_back(qpIdx);
                AllT4s_idxs_all.push_back(qpIdx);
            }
        }

        // Set the T4x idxs variables
        for (unsigned int it4x = 0; it4x < T4x_types.size(); ++it4x)
        {
            ana.tx.pushbackToBranch<vector<int>>(TString::Format("mtv_match_idxs_T4x_%s", T4x_types.at(it4x).Data()), T4x_idxs.at(it4x));
        }
        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_T4x_AllTypes", T4x_idxs_all);

        // Any T4s (not including pT4)
        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_AllT4s_AllTypes", AllT4s_idxs_all);

        // Triplet (T3)
        std::array<std::vector<int>, n_T3_types> T3_idxs;
        std::vector<int> T3_idxs_all;
        for (auto& tpIdx : sdl.sim_tpIdx()[isim])
        {
            std::vector<int> layers = {sdl.tp_layer()[tpIdx][0], sdl.tp_layer()[tpIdx][2], sdl.tp_layer()[tpIdx][4]};
            if (T3_types_map.find(layers) != T3_types_map.end())
            {
                T3_idxs[T3_types_map.at(layers)].push_back(tpIdx);
                T3_idxs_all.push_back(tpIdx);
            }
        }

        // Set the T3 idxs variables
        for (unsigned int it3 = 0; it3 < T3_types.size(); ++it3)
        {
            ana.tx.pushbackToBranch<vector<int>>(TString::Format("mtv_match_idxs_T3_%s", T3_types.at(it3).Data()), T3_idxs.at(it3));
        }
        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_T3_AllTypes", T3_idxs_all);

        // Track Candidates (TC)
        std::array<std::vector<int>, n_TC_types> TC_idxs;
        std::vector<int> TC_idxs_all;
        std::vector<int> TC_set1_idxs_all;
        std::vector<int> TC_set2_idxs_all;
        std::vector<int> TC_set3_idxs_all;
        std::vector<int> TC_set4_idxs_all;
        for (auto& tcIdx : sdl.sim_tcIdx()[isim])
        {
            const std::vector<int>& layers = sdl.tc_layer()[tcIdx];
            if (TC_types_map.find(layers) != TC_types_map.end())
            {
                TC_idxs[TC_types_map.at(layers)].push_back(tcIdx);
                TC_idxs_all.push_back(tcIdx);
            }
            if (TC_set1_types_map.find(layers) != TC_set1_types_map.end())
            {
                TC_set1_idxs_all.push_back(tcIdx);
            }
            if (TC_set2_types_map.find(layers) != TC_set2_types_map.end())
            {
                TC_set2_idxs_all.push_back(tcIdx);
            }
            if (TC_set3_types_map.find(layers) != TC_set3_types_map.end())
            {
                TC_set3_idxs_all.push_back(tcIdx);
            }
            if (TC_set4_types_map.find(layers) != TC_set4_types_map.end())
            {
                TC_set4_idxs_all.push_back(tcIdx);
            }
        }

        // Set the TC idxs variables
        for (unsigned int itc = 0; itc < TC_types.size(); ++itc)
        {
            ana.tx.pushbackToBranch<vector<int>>(TString::Format("mtv_match_idxs_TC_%s", TC_types.at(itc).Data()), TC_idxs.at(itc));
        }
        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_TC_AllTypes", TC_idxs_all);
        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_TC_Set1Types", TC_set1_idxs_all);
        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_TC_Set2Types", TC_set2_idxs_all);
        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_TC_Set3Types", TC_set3_idxs_all);
        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_TC_Set4Types", TC_set4_idxs_all);

        // Pixel Line segments
        std::vector<int> pix_idxs;
        if (std::find(unique_simtrkIdx_matched_to_seeds.begin(), unique_simtrkIdx_matched_to_seeds.end(), isim) != unique_simtrkIdx_matched_to_seeds.end())
        {
            pix_idxs.push_back(1);
        }

        ana.tx.pushbackToBranch<vector<int>>("mtv_match_idxs_pix_P", pix_idxs);

    }


}

void printSDLVariables()
{
    for (unsigned int isimtrk = 0; isimtrk < sdl.sim_pt().size(); ++isimtrk)
        printSDLVariablesForATrack(isimtrk);
}

void printSDLVariablesForATrack(int isimtrk)
{

    // hasAll12Hits
    const int& hasAll12Hits = ana.tx.getBranch<vector<int>>("sim_hasAll12HitsInBarrel")[isimtrk];

    // Mini-Doublets
    bool hasMD_B1 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_B1")[isimtrk].size() > 0;
    bool hasMD_B2 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_B2")[isimtrk].size() > 0;
    bool hasMD_B3 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_B3")[isimtrk].size() > 0;
    bool hasMD_B4 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_B4")[isimtrk].size() > 0;
    bool hasMD_B5 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_B5")[isimtrk].size() > 0;
    bool hasMD_B6 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_B6")[isimtrk].size() > 0;
    bool hasMD_E1 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_E1")[isimtrk].size() > 0;
    bool hasMD_E2 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_E2")[isimtrk].size() > 0;
    bool hasMD_E3 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_E3")[isimtrk].size() > 0;
    bool hasMD_E4 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_E4")[isimtrk].size() > 0;
    bool hasMD_E5 = ana.tx.getBranch<vector<vector<int>>>("mtv_match_idxs_MD_E5")[isimtrk].size() > 0;

    const float& pt = sdl.sim_pt()[isimtrk];
    const float& eta = sdl.sim_eta()[isimtrk];
    const float& dz = sdl.sim_pca_dz()[isimtrk];
    const float& dxy = sdl.sim_pca_dxy()[isimtrk];
    const float& phi = sdl.sim_phi()[isimtrk];
    // const int& bunch = sdl.sim_bunchCrossing()[isimtrk];
    const int& pdgid = sdl.sim_pdgId()[isimtrk];

    std::cout << "isimtrk : " << isimtrk << std::endl;
    std::cout <<  " pt: " << pt <<  " eta: " << eta <<  " pdgid: " << pdgid <<  " dz: " << dz <<  " dxy: " << dxy <<  " phi: " << phi <<  std::endl;
    std::cout << "hasAll12Hits : " << hasAll12Hits << std::endl;
    std::cout <<  " hasMD_B1: " << hasMD_B1 <<  " hasMD_B2: " << hasMD_B2 <<  " hasMD_B3: " << hasMD_B3 <<  " hasMD_B4: " << hasMD_B4 <<  " hasMD_B5: " << hasMD_B5 <<  " hasMD_B6: " << hasMD_B6 <<  " hasMD_E1: " << hasMD_E1 <<  " hasMD_E2: " << hasMD_E2 <<  " hasMD_E3: " << hasMD_E3 <<  " hasMD_E4: " << hasMD_E4 <<  " hasMD_E5: " << hasMD_E5 <<  std::endl;

}

void bookEfficiencySets(std::vector<EfficiencySetDefinition>& effsets)
{
    for (auto& effset : effsets)
        bookEfficiencySet(effset);
}

void bookEfficiencySet(EfficiencySetDefinition& effset)
{

    std::vector<float> pt_boundaries = getPtBounds();

    TString category_name = effset.set_name;

    // Denominator tracks' quantities
    ana.tx.createBranch<vector<float>>(category_name + "_denom_pt");
    ana.tx.createBranch<vector<float>>(category_name + "_denom_eta");
    ana.tx.createBranch<vector<float>>(category_name + "_denom_dxy");
    ana.tx.createBranch<vector<float>>(category_name + "_denom_dz");
    ana.tx.createBranch<vector<float>>(category_name + "_denom_phi");

    // Numerator tracks' quantities
    ana.tx.createBranch<vector<float>>(category_name + "_numer_pt");
    ana.tx.createBranch<vector<float>>(category_name + "_numer_eta");
    ana.tx.createBranch<vector<float>>(category_name + "_numer_dxy");
    ana.tx.createBranch<vector<float>>(category_name + "_numer_dz");
    ana.tx.createBranch<vector<float>>(category_name + "_numer_phi");

    // Histogram utility object that is used to define the histograms
    ana.histograms.addVecHistogram(category_name + "_h_denom_pt"  , pt_boundaries     , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_denom_pt"); } );
    ana.histograms.addVecHistogram(category_name + "_h_numer_pt"  , pt_boundaries     , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_numer_pt"); } );
    ana.histograms.addVecHistogram(category_name + "_h_denom_eta" , 180 , -2.5  , 2.5  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_denom_eta"); } );
    ana.histograms.addVecHistogram(category_name + "_h_numer_eta" , 180 , -2.5  , 2.5  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_numer_eta"); } );
    ana.histograms.addVecHistogram(category_name + "_h_denom_dxy" , 180 , -30.  , 30.  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_denom_dxy"); } );
    ana.histograms.addVecHistogram(category_name + "_h_numer_dxy" , 180 , -30.  , 30.  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_numer_dxy"); } );
    ana.histograms.addVecHistogram(category_name + "_h_denom_dz"  , 180 , -30.  , 30.  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_denom_dz"); } );
    ana.histograms.addVecHistogram(category_name + "_h_numer_dz"  , 180 , -30.  , 30.  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_numer_dz"); } );
    ana.histograms.addVecHistogram(category_name + "_h_denom_phi" , 180 , -M_PI , M_PI , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_denom_phi"); } );
    ana.histograms.addVecHistogram(category_name + "_h_numer_phi" , 180 , -M_PI , M_PI , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_numer_phi"); } );

}

void fillEfficiencySets(std::vector<EfficiencySetDefinition>& effsets)
{
    for (unsigned int isimtrk = 0; isimtrk < sdl.sim_pt().size(); ++isimtrk)
    {
        for (auto& effset : effsets)
        {
            fillEfficiencySet(isimtrk, effset);
        }
    }
}

void fillEfficiencySet(int isimtrk, EfficiencySetDefinition& effset)
{
    const float& pt = sdl.sim_pt()[isimtrk];
    const float& eta = sdl.sim_eta()[isimtrk];
    const float& dz = sdl.sim_pca_dz()[isimtrk];
    const float& dxy = sdl.sim_pca_dxy()[isimtrk];
    const float& phi = sdl.sim_phi()[isimtrk];
    const int& bunch = sdl.sim_bunchCrossing()[isimtrk];
    const int& event = sdl.sim_event()[isimtrk];
    const int& vtxIdx = sdl.sim_parentVtxIdx()[isimtrk];
    const int& pdgidtrk = sdl.sim_pdgId()[isimtrk];
    const int& q = sdl.sim_q()[isimtrk];
    const float& vtx_x = sdl.simvtx_x()[vtxIdx];
    const float& vtx_y = sdl.simvtx_y()[vtxIdx];
    const float& vtx_z = sdl.simvtx_z()[vtxIdx];
    const float& vtx_perp = sqrt(vtx_x * vtx_x + vtx_y * vtx_y);

    if (bunch != 0)
        return;

    if (event != 0)
        return;

    // if (vtxIdx != 0)
    //     return;

    if (ana.pdgid != 0 and abs(pdgidtrk) != abs(ana.pdgid))
        return;

    if (ana.pdgid == 0 and q == 0)
        return;

    // if (sdl.sim_psgIdx()[isimtrk].size() > 0)
    // {
    //     if (sdl.sim_pqpIdx()[isimtrk].size() == 0)
    //     {
    //         // std::cout <<  std::endl;
    //         // std::cout <<  " ana.looper.getCurrentEventIndex(): " << ana.looper.getCurrentEventIndex() <<  std::endl;
    //         // std::cout <<  " isimtrk: " << isimtrk <<  std::endl;
    //         int n_OT_hits = 0;
    //         int n_OT_hits_close = 0;
    //         SDLMath::Helix helix(pt, eta, phi, sdl.simvtx_x()[vtxIdx], sdl.simvtx_y()[vtxIdx], sdl.simvtx_z()[vtxIdx], sdl.sim_q()[isimtrk]);
    //         for (auto& isimhit : sdl.sim_simHitIdx()[isimtrk])
    //         {
    //             const auto& subdet = sdl.simhit_subdet()[isimhit];
    //             if (subdet == 4 or subdet == 5)
    //             {
    //                 n_OT_hits++;
    //                 std::vector<float> point = {sdl.simhit_x()[isimhit], sdl.simhit_y()[isimhit], sdl.simhit_z()[isimhit]};
    //                 float r = sqrt(point[0] * point[0] + point[1] * point[1]);
    //                 // std::cout <<  " r: " << r <<  std::endl;
    //                 // std::cout <<  " point[2]: " << point[2] <<  std::endl;
    //                 float r_diff = helix.compare_radius(point);
    //                 float xy_diff = helix.compare_xy(point);
    //                 // std::cout <<  " r_diff: " << r_diff <<  " xy_diff: " << xy_diff <<  std::endl;
    //                 if (xy_diff < 1.0)
    //                 {
    //                     n_OT_hits_close++;
    //                 }
    //             }

    //         }
    //         // std::cout <<  " n_OT_hits: " << n_OT_hits <<  std::endl;
    //         // std::cout <<  " n_OT_hits_close: " << n_OT_hits_close <<  std::endl;
    //         if (n_OT_hits_close < 4)
    //             return;
    //     }
    // }

    TString category_name = effset.set_name;

    // https://github.com/cms-sw/cmssw/blob/7cbdb18ec6d11d5fd17ca66c1153f0f4e869b6b0/SimTracker/Common/python/trackingParticleSelector_cfi.py
    // https://github.com/cms-sw/cmssw/blob/7cbdb18ec6d11d5fd17ca66c1153f0f4e869b6b0/SimTracker/Common/interface/TrackingParticleSelector.h#L122-L124
    const float vtx_z_thresh = 30;
    const float vtx_perp_thresh = 2.5;

    if (pt > 1.5 and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh)
        ana.tx.pushbackToBranch<float>(category_name + "_denom_eta", eta);
    if (abs(eta) < 2.4 and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh)
        ana.tx.pushbackToBranch<float>(category_name + "_denom_pt", pt);
    if (abs(eta) < 2.4 and pt > 1.5 and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh)
        ana.tx.pushbackToBranch<float>(category_name + "_denom_phi", phi);
    if (abs(eta) < 2.4 and pt > 1.5 and abs(vtx_z) < vtx_z_thresh)
        ana.tx.pushbackToBranch<float>(category_name + "_denom_dxy", dxy);
    if (abs(eta) < 2.4 and pt > 1.5 and abs(vtx_perp) < vtx_perp_thresh)
        ana.tx.pushbackToBranch<float>(category_name + "_denom_dz", dz);

    if (effset.pass(isimtrk))
    {
        if (pt > 1.5 and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh)
            ana.tx.pushbackToBranch<float>(category_name + "_numer_eta", eta);
        if (abs(eta) < 2.4 and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh)
            ana.tx.pushbackToBranch<float>(category_name + "_numer_pt", pt);
        if (abs(eta) < 2.4 and pt > 1.5 and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh)
            ana.tx.pushbackToBranch<float>(category_name + "_numer_phi", phi);
        if (abs(eta) < 2.4 and pt > 1.5 and abs(vtx_z) < vtx_z_thresh)
            ana.tx.pushbackToBranch<float>(category_name + "_numer_dxy", dxy);
        if (abs(eta) < 2.4 and pt > 1.5 and abs(vtx_perp) < vtx_perp_thresh)
            ana.tx.pushbackToBranch<float>(category_name + "_numer_dz", dz);
    }
}

