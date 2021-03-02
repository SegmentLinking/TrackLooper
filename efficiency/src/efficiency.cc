#include "efficiency.h"
#include "SDLMath.h"

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

    list_effSetDef.push_back(EfficiencySetDefinition("TC_AllTypes", 13, [&](int isim) {return sdl.sim_TC_matched()[isim] > 0;}));
    if (ana.do_lower_level)
    {
        list_effSetDef.push_back(EfficiencySetDefinition("T4s_AllTypes", 13, [&](int isim) {return sdl.sim_T4_matched()[isim] > 0;}));
        list_effSetDef.push_back(EfficiencySetDefinition("T3_AllTypes", 13, [&](int isim) {return sdl.sim_T3_matched()[isim] > 0;}));
    }

    bookEfficiencySets(list_effSetDef);

    // creating a set of fake rate plots
    std::vector<FakeRateSetDefinition> list_FRSetDef;

    list_FRSetDef.push_back(FakeRateSetDefinition("TC_AllTypes", 13, [&](int itc) {return sdl.tc_isFake()[itc] > 0;}, sdl.tc_pt(), sdl.tc_eta(), sdl.tc_phi()));
    if (ana.do_lower_level)
    {
        list_FRSetDef.push_back(FakeRateSetDefinition("T4s_AllTypes", 13, [&](int it4) {return sdl.t4_isFake()[it4] > 0;}, sdl.t4_pt(), sdl.t4_eta(), sdl.t4_phi()));
        list_FRSetDef.push_back(FakeRateSetDefinition("T3_AllTypes", 13, [&](int it3) {return sdl.t3_isFake()[it3] > 0;}, sdl.t3_pt(), sdl.t3_eta(), sdl.t3_phi()));
        list_FRSetDef.push_back(FakeRateSetDefinition("pLS_AllTypes", 13, [&](int itls) { return sdl.pLS_isFake()[itls] > 0;}, sdl.pLS_pt(), sdl.pLS_eta(), sdl.pLS_phi()));
    }

    bookFakeRateSets(list_FRSetDef);

    // creating a set of fake rate plots
    std::vector<DuplicateRateSetDefinition> list_DLSetDef;

    list_DLSetDef.push_back(DuplicateRateSetDefinition("TC_AllTypes", 13, [&](int itc) {return sdl.tc_isDuplicate()[itc] > 0;}, sdl.tc_pt(), sdl.tc_eta(), sdl.tc_phi()));
    if (ana.do_lower_level)
    {
        list_DLSetDef.push_back(DuplicateRateSetDefinition("T4s_AllTypes", 13, [&](int it4) {return sdl.t4_isDuplicate()[it4] > 0;}, sdl.t4_pt(), sdl.t4_eta(), sdl.t4_phi()));
        list_DLSetDef.push_back(DuplicateRateSetDefinition("T3_AllTypes", 13, [&](int it3) {return sdl.t3_isDuplicate()[it3] > 0;}, sdl.t3_pt(), sdl.t3_eta(), sdl.t3_phi()));
        list_DLSetDef.push_back(DuplicateRateSetDefinition("pLS_AllTypes", 13, [&](int itls) { return sdl.pLS_isDuplicate()[itls] > 0;}, sdl.pLS_pt(), sdl.pLS_eta(), sdl.pLS_phi()));
    }

    bookDuplicateRateSets(list_DLSetDef);

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
        fillFakeRateSets(list_FRSetDef);
        fillDuplicateRateSets(list_DLSetDef);

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
}

void setSDLVariables()
{
}

void printSDLVariables()
{
}

void printSDLVariablesForATrack(int isimtrk)
{
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
    for (auto& effset : effsets)
    {
        for (unsigned int isimtrk = 0; isimtrk < sdl.sim_pt().size(); ++isimtrk)
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

    if (ana.pdgid != 0 and abs(pdgidtrk) != abs(ana.pdgid))
        return;

    if (ana.pdgid == 0 and q == 0)
        return;

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

void bookFakeRateSets(std::vector<FakeRateSetDefinition>& FRsets)
{
    for (auto& FRset : FRsets)
        bookFakeRateSet(FRset);
}

void bookFakeRateSet(FakeRateSetDefinition& FRset)
{

    std::vector<float> pt_boundaries = getPtBounds();

    TString category_name = FRset.set_name;

    // Denominator tracks' quantities
    ana.tx.createBranch<vector<float>>(category_name + "_fakerate_denom_pt");
    ana.tx.createBranch<vector<float>>(category_name + "_fakerate_denom_eta");
    ana.tx.createBranch<vector<float>>(category_name + "_fakerate_denom_phi");

    // Numerator tracks' quantities
    ana.tx.createBranch<vector<float>>(category_name + "_fakerate_numer_pt");
    ana.tx.createBranch<vector<float>>(category_name + "_fakerate_numer_eta");
    ana.tx.createBranch<vector<float>>(category_name + "_fakerate_numer_phi");

    // Histogram utility object that is used to define the histograms
    ana.histograms.addVecHistogram(category_name + "_h_fakerate_denom_pt"  , pt_boundaries     , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_fakerate_denom_pt"); } );
    ana.histograms.addVecHistogram(category_name + "_h_fakerate_numer_pt"  , pt_boundaries     , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_fakerate_numer_pt"); } );
    ana.histograms.addVecHistogram(category_name + "_h_fakerate_denom_eta" , 180 , -2.5  , 2.5  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_fakerate_denom_eta"); } );
    ana.histograms.addVecHistogram(category_name + "_h_fakerate_numer_eta" , 180 , -2.5  , 2.5  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_fakerate_numer_eta"); } );
    ana.histograms.addVecHistogram(category_name + "_h_fakerate_denom_phi" , 180 , -M_PI , M_PI , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_fakerate_denom_phi"); } );
    ana.histograms.addVecHistogram(category_name + "_h_fakerate_numer_phi" , 180 , -M_PI , M_PI , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_fakerate_numer_phi"); } );

}

void fillFakeRateSets(std::vector<FakeRateSetDefinition>& FRsets)
{
    for (auto& FRset : FRsets)
    {
        for (unsigned int itc = 0; itc < sdl.tc_pt().size(); ++itc)
        {
            fillFakeRateSet(itc, FRset);
        }
    }
}

void fillFakeRateSet(int itc, FakeRateSetDefinition& FRset)
{
    const float& pt = FRset.pt[itc];
    const float& eta = FRset.eta[itc];
    const float& phi = FRset.phi[itc];

    TString category_name = FRset.set_name;

    if (pt > 1.5)
        ana.tx.pushbackToBranch<float>(category_name + "_fakerate_denom_eta", eta);
    if (abs(eta) < 2.4)
        ana.tx.pushbackToBranch<float>(category_name + "_fakerate_denom_pt", pt);
    if (abs(eta) < 2.4 and pt > 1.5)
        ana.tx.pushbackToBranch<float>(category_name + "_fakerate_denom_phi", phi);

    if (FRset.pass(itc))
    {
        if (pt > 1.5)
            ana.tx.pushbackToBranch<float>(category_name + "_fakerate_numer_eta", eta);
        if (abs(eta) < 2.4)
            ana.tx.pushbackToBranch<float>(category_name + "_fakerate_numer_pt", pt);
        if (abs(eta) < 2.4 and pt > 1.5)
            ana.tx.pushbackToBranch<float>(category_name + "_fakerate_numer_phi", phi);
    }
}

void bookDuplicateRateSets(std::vector<DuplicateRateSetDefinition>& DLsets)
{
    for (auto& DLset : DLsets)
        bookDuplicateRateSet(DLset);
}

void bookDuplicateRateSet(DuplicateRateSetDefinition& DLset)
{

    std::vector<float> pt_boundaries = getPtBounds();

    TString category_name = DLset.set_name;

    // Denominator tracks' quantities
    ana.tx.createBranch<vector<float>>(category_name + "_duplrate_denom_pt");
    ana.tx.createBranch<vector<float>>(category_name + "_duplrate_denom_eta");
    ana.tx.createBranch<vector<float>>(category_name + "_duplrate_denom_phi");

    // Numerator tracks' quantities
    ana.tx.createBranch<vector<float>>(category_name + "_duplrate_numer_pt");
    ana.tx.createBranch<vector<float>>(category_name + "_duplrate_numer_eta");
    ana.tx.createBranch<vector<float>>(category_name + "_duplrate_numer_phi");

    // Histogram utility object that is used to define the histograms
    ana.histograms.addVecHistogram(category_name + "_h_duplrate_denom_pt"  , pt_boundaries     , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_duplrate_denom_pt"); } );
    ana.histograms.addVecHistogram(category_name + "_h_duplrate_numer_pt"  , pt_boundaries     , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_duplrate_numer_pt"); } );
    ana.histograms.addVecHistogram(category_name + "_h_duplrate_denom_eta" , 180 , -2.5  , 2.5  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_duplrate_denom_eta"); } );
    ana.histograms.addVecHistogram(category_name + "_h_duplrate_numer_eta" , 180 , -2.5  , 2.5  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_duplrate_numer_eta"); } );
    ana.histograms.addVecHistogram(category_name + "_h_duplrate_denom_phi" , 180 , -M_PI , M_PI , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_duplrate_denom_phi"); } );
    ana.histograms.addVecHistogram(category_name + "_h_duplrate_numer_phi" , 180 , -M_PI , M_PI , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_duplrate_numer_phi"); } );

}

void fillDuplicateRateSets(std::vector<DuplicateRateSetDefinition>& DLsets)
{
    for (auto& DLset : DLsets)
    {
        for (unsigned int itc = 0; itc < sdl.tc_pt().size(); ++itc)
        {
            fillDuplicateRateSet(itc, DLset);
        }
    }
}

void fillDuplicateRateSet(int itc, DuplicateRateSetDefinition& DLset)
{
    const float& pt = DLset.pt[itc];
    const float& eta = DLset.eta[itc];
    const float& phi = DLset.phi[itc];

    TString category_name = DLset.set_name;

    if (pt > 1.5)
        ana.tx.pushbackToBranch<float>(category_name + "_duplrate_denom_eta", eta);
    if (abs(eta) < 2.4)
        ana.tx.pushbackToBranch<float>(category_name + "_duplrate_denom_pt", pt);
    if (abs(eta) < 2.4 and pt > 1.5)
        ana.tx.pushbackToBranch<float>(category_name + "_duplrate_denom_phi", phi);

    if (DLset.pass(itc))
    {
        if (pt > 1.5)
            ana.tx.pushbackToBranch<float>(category_name + "_duplrate_numer_eta", eta);
        if (abs(eta) < 2.4)
            ana.tx.pushbackToBranch<float>(category_name + "_duplrate_numer_pt", pt);
        if (abs(eta) < 2.4 and pt > 1.5)
            ana.tx.pushbackToBranch<float>(category_name + "_duplrate_numer_phi", phi);
    }
}

