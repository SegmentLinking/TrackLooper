#include "efficiency.h"
#include "SDLMath.h"

#include "sdl_types.h"

#define PTCUT 0.9 
#define ETACUT 4.5
// #define PTCUT 1.5
// #define ETACUT 2.4

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

    list_effSetDef.push_back(EfficiencySetDefinition("TC_AllTypes", 13, [&](int isim) {return sdl.sim_TC_matched().size() > isim ? sdl.sim_TC_matched()[isim] > 0 : false;}));
    if (ana.do_lower_level)
    {
        list_effSetDef.push_back(EfficiencySetDefinition("T4s_AllTypes", 13, [&](int isim) {return sdl.sim_T4_matched().size() > isim ? sdl.sim_T4_matched()[isim] > 0 : false;}));
        list_effSetDef.push_back(EfficiencySetDefinition("T3_AllTypes", 13, [&](int isim) {return sdl.sim_T3_matched().size() > isim ? sdl.sim_T3_matched()[isim] > 0 : false;}));
        list_effSetDef.push_back(EfficiencySetDefinition("pT4_AllTypes", 13, [&](int isim) {return sdl.sim_pT4_matched().size() > isim ? sdl.sim_pT4_matched()[isim] > 0 : false;}));
        list_effSetDef.push_back(EfficiencySetDefinition("pLS_AllTypes", 13, [&](int isim) {return sdl.sim_pLS_matched().size() > isim ? sdl.sim_pLS_matched()[isim] > 0 : false;}));
        list_effSetDef.push_back(EfficiencySetDefinition("T5_AllTypes", 13, [&](int isim) {return sdl.sim_T5_matched().size() > isim ? sdl.sim_T5_matched()[isim] > 0 : false;}));
        list_effSetDef.push_back(EfficiencySetDefinition("pT3_AllTypes", 13, [&](int isim) {return sdl.sim_pT3_matched().size() > isim ? sdl.sim_pT3_matched()[isim] > 0 : false;}));
        list_effSetDef.push_back(EfficiencySetDefinition("pT5_AllTypes", 13, [&](int isim) {return sdl.sim_pT5_matched().size() > isim ? sdl.sim_pT5_matched()[isim] > 0 : false;}));
        list_effSetDef.push_back(EfficiencySetDefinition("TCE_AllTypes", 13, [&](int isim) {return sdl.sim_tce_matched().size() > isim ? sdl.sim_tce_matched()[isim] > 0 : false;}));
        list_effSetDef.push_back(EfficiencySetDefinition("pureTCE_AllTypes", 13, [&](int isim) {return sdl.sim_pureTCE_matched().size() > isim ? sdl.sim_pureTCE_matched()[isim] > 0 : false;}));
#ifdef T3T3_EXTENSIONS
        list_effSetDef.push_back(EfficiencySetDefinition("T3T3_AllTypes", 13, [&](int isim) {return sdl.sim_T3T3_matched().size() > isim ? sdl.sim_T3T3_matched()[isim] > 0 : false;}));
#endif
    }

    bookEfficiencySets(list_effSetDef);

    // creating a set of fake rate plots
    std::vector<FakeRateSetDefinition> list_FRSetDef;

    // TODO FIXME the passing of the pt eta phi's are not implemented properly. (It's like half finished....)
    list_FRSetDef.push_back(FakeRateSetDefinition("TC_AllTypes", 13, [&](int itc) {return sdl.tc_isFake().size() > itc ? sdl.tc_isFake()[itc] > 0 : false;}, sdl.tc_pt(), sdl.tc_eta(), sdl.tc_phi()));
    if (ana.do_lower_level)
    {
        list_FRSetDef.push_back(FakeRateSetDefinition("T4s_AllTypes", 13, [&](int it4) {return sdl.t4_isFake().size() > it4 ? sdl.t4_isFake()[it4] > 0 : false;}, sdl.t4_pt(), sdl.t4_eta(), sdl.t4_phi()));
        list_FRSetDef.push_back(FakeRateSetDefinition("T3_AllTypes", 13, [&](int it3) {return sdl.t3_isFake().size() > it3 ? sdl.t3_isFake()[it3] > 0 : false;}, sdl.t3_pt(), sdl.t3_eta(), sdl.t3_phi()));
        list_FRSetDef.push_back(FakeRateSetDefinition("pT4_AllTypes", 13, [&](int ipT4) {return sdl.pT4_isFake().size() > ipT4 ? sdl.pT4_isFake()[ipT4] > 0 : false;}, sdl.pT4_pt(), sdl.pT4_eta(), sdl.pT4_phi()));
        list_FRSetDef.push_back(FakeRateSetDefinition("pLS_AllTypes", 13, [&](int itls) {return sdl.pLS_isFake().size() > itls ? sdl.pLS_isFake()[itls] > 0 : false;}, sdl.pLS_pt(), sdl.pLS_eta(), sdl.pLS_phi()));
        list_FRSetDef.push_back(FakeRateSetDefinition("T5_AllTypes", 13, [&](int iT5) {return sdl.t5_isFake().size() > iT5 ? sdl.t5_isFake()[iT5] > 0 : false;}, sdl.t5_pt(), sdl.t5_eta(), sdl.t5_phi()));
        list_FRSetDef.push_back(FakeRateSetDefinition("pT3_AllTypes", 13, [&](int ipT3) {return sdl.pT3_isFake().size() > ipT3 ? sdl.pT3_isFake()[ipT3] > 0 : false;}, sdl.pT3_pt(), sdl.pT3_eta(), sdl.pT3_phi()));
        list_FRSetDef.push_back(FakeRateSetDefinition("pT5_AllTypes", 13, [&](int ipT5) {return sdl.pT5_isFake().size() > ipT5 ? sdl.pT5_isFake()[ipT5] > 0 : false;}, sdl.pT5_pt(), sdl.pT5_eta(), sdl.pT5_phi()));
        list_FRSetDef.push_back(FakeRateSetDefinition("TCE_AllTypes", 13, [&](int iTCE) {return sdl.tce_isFake().size() > iTCE ? sdl.tce_isFake()[iTCE] > 0 : false;}, sdl.tce_pt(), sdl.tce_eta(), sdl.tce_phi()));
        list_FRSetDef.push_back(FakeRateSetDefinition("pureTCE_AllTypes", 13, [&](int ipureTCE) {return sdl.tce_isFake().size() > ipureTCE ? sdl.tce_isFake()[ipureTCE] > 0 : false;}, sdl.pureTCE_pt(), sdl.pureTCE_eta(), sdl.pureTCE_phi()));
#ifdef T3T3_EXTENSIONS
        list_FRSetDef.push_back(FakeRateSetDefinition("T3T3_AllTypes", 13, [&](int iT3T3) {return sdl.T3T3_isFake().size() > iT3T3 ? sdl.T3T3_isFake()[iT3T3] > 0 : false;}, sdl.T3T3_pt(), sdl.T3T3_eta(), sdl.T3T3_phi()));
#endif
    }

    bookFakeRateSets(list_FRSetDef);

    // creating a set of fake rate plots
    std::vector<DuplicateRateSetDefinition> list_DLSetDef;

    list_DLSetDef.push_back(DuplicateRateSetDefinition("TC_AllTypes", 13, [&](int itc) {return sdl.tc_isDuplicate().size() > itc ? sdl.tc_isDuplicate()[itc] > 0 : false;}, sdl.tc_pt(), sdl.tc_eta(), sdl.tc_phi()));
    if (ana.do_lower_level)
    {
        list_DLSetDef.push_back(DuplicateRateSetDefinition("T4s_AllTypes", 13, [&](int it4) {return sdl.t4_isDuplicate().size() > it4 ? sdl.t4_isDuplicate()[it4] > 0 : false;}, sdl.t4_pt(), sdl.t4_eta(), sdl.t4_phi()));
        list_DLSetDef.push_back(DuplicateRateSetDefinition("T3_AllTypes", 13, [&](int it3) {return sdl.t3_isDuplicate().size() > it3 ? sdl.t3_isDuplicate()[it3] > 0 : false;}, sdl.t3_pt(), sdl.t3_eta(), sdl.t3_phi()));
        list_DLSetDef.push_back(DuplicateRateSetDefinition("pT4_AllTypes", 13, [&](int ipT4) {return sdl.pT4_isDuplicate().size() > ipT4 ? sdl.pT4_isDuplicate()[ipT4] > 0 : false;}, sdl.pT4_pt(), sdl.pT4_eta(), sdl.pT4_phi()));
        list_DLSetDef.push_back(DuplicateRateSetDefinition("pLS_AllTypes", 13, [&](int itls) {return sdl.pLS_isDuplicate().size() > itls ? sdl.pLS_isDuplicate()[itls] > 0 : false;}, sdl.pLS_pt(), sdl.pLS_eta(), sdl.pLS_phi()));
        list_DLSetDef.push_back(DuplicateRateSetDefinition("T5_AllTypes", 13, [&](int iT5) {return sdl.t5_isDuplicate().size() > iT5 ? sdl.t5_isDuplicate()[iT5] > 0 : false;}, sdl.t5_pt(), sdl.t5_eta(), sdl.t5_phi()));
        list_DLSetDef.push_back(DuplicateRateSetDefinition("pT3_AllTypes", 13, [&](int ipT3) {return sdl.pT3_isDuplicate().size() > ipT3 ? sdl.pT3_isDuplicate()[ipT3] > 0 : false;}, sdl.pT3_pt(), sdl.pT3_eta(), sdl.pT3_phi()));
        list_DLSetDef.push_back(DuplicateRateSetDefinition("pT5_AllTypes", 13, [&](int ipT5) {return sdl.pT5_isDuplicate().size() > ipT5 ? sdl.pT5_isDuplicate()[ipT5] > 0 : false;}, sdl.pT5_pt(), sdl.pT5_eta(), sdl.pT5_phi()));
        list_DLSetDef.push_back(DuplicateRateSetDefinition("TCE_AllTypes", 13, [&](int iTCE) {return sdl.tce_isDuplicate().size() > iTCE ? sdl.tce_isDuplicate()[iTCE] > 0 : false;}, sdl.tce_pt(), sdl.tce_eta(), sdl.tce_phi()));  
        list_DLSetDef.push_back(DuplicateRateSetDefinition("pureTCE_AllTypes", 13, [&](int ipureTCE) {return sdl.tce_isDuplicate().size() > ipureTCE ? sdl.tce_isDuplicate()[ipureTCE] > 0 : false;}, sdl.pureTCE_pt(), sdl.pureTCE_eta(), sdl.pureTCE_phi()));   
#ifdef T3T3_EXTENSIONS
        list_DLSetDef.push_back(DuplicateRateSetDefinition("T3T3_AllTypes", 13, [&](int iT3T3) {return sdl.T3T3_isDuplicate().size() > iT3T3 ? sdl.T3T3_isDuplicate()[iT3T3] > 0 : false;}, sdl.T3T3_pt(), sdl.T3T3_eta(), sdl.T3T3_phi()));   
#endif

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
        // std::cout <<  " 'here': " << "here" <<  std::endl;
        // std::cout <<  " list_FRSetDef.size(): " << list_FRSetDef.size() <<  std::endl;
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
    ana.tx.createBranch<vector<float>>(category_name + "_denom_hits");
    ana.tx.createBranch<vector<float>>(category_name + "_denom_layers");
    ana.tx.createBranch<vector<float>>(category_name + "_denom_layersgap");

    // Numerator tracks' quantities
    ana.tx.createBranch<vector<float>>(category_name + "_numer_pt");
    ana.tx.createBranch<vector<float>>(category_name + "_numer_eta");
    ana.tx.createBranch<vector<float>>(category_name + "_numer_dxy");
    ana.tx.createBranch<vector<float>>(category_name + "_numer_dz");
    ana.tx.createBranch<vector<float>>(category_name + "_numer_phi");
    ana.tx.createBranch<vector<float>>(category_name + "_numer_hits");
    ana.tx.createBranch<vector<float>>(category_name + "_numer_layers");
    ana.tx.createBranch<vector<float>>(category_name + "_numer_layersgap");
    //inefficiencies
    // Denominator tracks' quantities
    ana.tx.createBranch<vector<float>>(category_name + "_inefficiency_denom_pt");
    ana.tx.createBranch<vector<float>>(category_name + "_inefficiency_denom_eta");
    ana.tx.createBranch<vector<float>>(category_name + "_inefficiency_denom_dxy");
    ana.tx.createBranch<vector<float>>(category_name + "_inefficiency_denom_dz");
    ana.tx.createBranch<vector<float>>(category_name + "_inefficiency_denom_phi");
    ana.tx.createBranch<vector<float>>(category_name + "_inefficiency_denom_hits");
    ana.tx.createBranch<vector<float>>(category_name + "_inefficiency_denom_layers");
    ana.tx.createBranch<vector<float>>(category_name + "_inefficiency_denom_layersgap");

    ana.tx.createBranch<vector<float>>(category_name + "_inefficiency_numer_pt");
    ana.tx.createBranch<vector<float>>(category_name + "_inefficiency_numer_eta");
    ana.tx.createBranch<vector<float>>(category_name + "_inefficiency_numer_dxy");
    ana.tx.createBranch<vector<float>>(category_name + "_inefficiency_numer_dz");
    ana.tx.createBranch<vector<float>>(category_name + "_inefficiency_numer_phi");
    ana.tx.createBranch<vector<float>>(category_name + "_inefficiency_numer_hits");
    ana.tx.createBranch<vector<float>>(category_name + "_inefficiency_numer_layers");
    ana.tx.createBranch<vector<float>>(category_name + "_inefficiency_numer_layersgap");


    // Histogram utility object that is used to define the histograms
    ana.histograms.addVecHistogram(category_name + "_h_denom_pt"  , pt_boundaries     , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_denom_pt"); } );
    ana.histograms.addVecHistogram(category_name + "_h_numer_pt"  , pt_boundaries     , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_numer_pt"); } );
    ana.histograms.addVecHistogram(category_name + "_h_denom_eta" , 180 , -4.5  , 4.5  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_denom_eta"); } );
    ana.histograms.addVecHistogram(category_name + "_h_numer_eta" , 180 , -4.5  , 4.5  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_numer_eta"); } );
    ana.histograms.addVecHistogram(category_name + "_h_denom_dxy" , 180 , -30.  , 30.  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_denom_dxy"); } );
    ana.histograms.addVecHistogram(category_name + "_h_numer_dxy" , 180 , -30.  , 30.  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_numer_dxy"); } );
    ana.histograms.addVecHistogram(category_name + "_h_denom_dz"  , 180 , -30.  , 30.  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_denom_dz"); } );
    ana.histograms.addVecHistogram(category_name + "_h_numer_dz"  , 180 , -30.  , 30.  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_numer_dz"); } );
    ana.histograms.addVecHistogram(category_name + "_h_denom_phi" , 180 , -M_PI , M_PI , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_denom_phi"); } );
    ana.histograms.addVecHistogram(category_name + "_h_numer_phi" , 180 , -M_PI , M_PI , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_numer_phi"); } );
    ana.histograms.addVecHistogram(category_name + "_h_denom_hits" , 30 , 0 , 30 , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_denom_hits"); } );
    ana.histograms.addVecHistogram(category_name + "_h_numer_hits" , 30 , 0 , 30 , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_numer_hits"); } );
    ana.histograms.addVecHistogram(category_name + "_h_denom_layers" , 9 , -2 , 7 , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_denom_layers"); } );
    ana.histograms.addVecHistogram(category_name + "_h_numer_layers" , 9 , -2 , 7 , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_numer_layers"); } );
    ana.histograms.addVecHistogram(category_name + "_h_denom_layersgap" , 9 , -2 , 7 , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_denom_layersgap"); } );
    ana.histograms.addVecHistogram(category_name + "_h_numer_layersgap" , 9 , -2 , 7 , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_numer_layersgap"); } );
    ana.histograms.addVecHistogram(category_name + "_h_inefficiency_denom_pt"  , pt_boundaries      , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_inefficiency_denom_pt"); } );
    ana.histograms.addVecHistogram(category_name + "_h_inefficiency_denom_eta" , 180 , -4.5  , 4.5  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_inefficiency_denom_eta"); } );
    ana.histograms.addVecHistogram(category_name + "_h_inefficiency_denom_dxy" , 180 , -30.  , 30.  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_inefficiency_denom_dxy"); } );
    ana.histograms.addVecHistogram(category_name + "_h_inefficiency_denom_dz"  , 180 , -30.  , 30.  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_inefficiency_denom_dz"); } );
    ana.histograms.addVecHistogram(category_name + "_h_inefficiency_denom_phi" , 180 , -M_PI , M_PI , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_inefficiency_denom_phi"); } );
    ana.histograms.addVecHistogram(category_name + "_h_inefficiency_numer_pt"  , pt_boundaries      , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_inefficiency_numer_pt"); } );
    ana.histograms.addVecHistogram(category_name + "_h_inefficiency_numer_eta" , 180 , -4.5  , 4.5  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_inefficiency_numer_eta"); } );
    ana.histograms.addVecHistogram(category_name + "_h_inefficiency_numer_dxy" , 180 , -30.  , 30.  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_inefficiency_numer_dxy"); } );
    ana.histograms.addVecHistogram(category_name + "_h_inefficiency_numer_dz"  , 180 , -30.  , 30.  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_inefficiency_numer_dz"); } );
    ana.histograms.addVecHistogram(category_name + "_h_inefficiency_numer_phi" , 180 , -M_PI , M_PI , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_inefficiency_numer_phi"); } );
    ana.histograms.addVecHistogram(category_name + "_h_inefficiency_denom_layers" , 9 , -2 , 7 , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_inefficiency_denom_layers"); } );
    ana.histograms.addVecHistogram(category_name + "_h_inefficiency_numer_layers" , 9 , -2 , 7 , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_inefficiency_numer_layers"); } );
    ana.histograms.addVecHistogram(category_name + "_h_inefficiency_denom_layersgap" , 9 , -2 , 7 , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_inefficiency_denom_layersgap"); } );
    ana.histograms.addVecHistogram(category_name + "_h_inefficiency_numer_layersgap" , 9 , -2 , 7 , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_inefficiency_numer_layersgap"); } );
    ana.histograms.addVecHistogram(category_name + "_h_inefficiency_denom_hits" , 30 , 0 , 30 , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_inefficiency_denom_hits"); } );
    ana.histograms.addVecHistogram(category_name + "_h_inefficiency_numer_hits" , 30 , 0 , 30 , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_inefficiency_numer_hits"); } );

}

void fillEfficiencySets(std::vector<EfficiencySetDefinition>& effsets)
{
    for (auto& effset : effsets)
    {
        bool excludepT5FoundSim = effset.set_name.Contains("pT3_");
        for (unsigned int isimtrk = 0; isimtrk < sdl.sim_pt().size(); ++isimtrk)
        {
            fillEfficiencySet(isimtrk, effset,excludepT5FoundSim);
        }
    }
}

void fillEfficiencySet(int isimtrk, EfficiencySetDefinition& effset, bool excludepT5Found)
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
    const float& lay = sdl.sim_len()[isimtrk];
    const float& laygap = sdl.sim_lengap()[isimtrk];
    const float& hits = sdl.sim_hits()[isimtrk];

    const float& pT5Found = sdl.sim_pT5_matched()[isimtrk];

    if (bunch != 0)
        return;

    if (event != 0)
        return;

    if (ana.pdgid != 0 and abs(pdgidtrk) != abs(ana.pdgid))
        return;

    if (ana.pdgid == 0 and q == 0)
        return;

    if(pT5Found !=0 and excludepT5Found)
      return;

    TString category_name = effset.set_name;

    // https://github.com/cms-sw/cmssw/blob/7cbdb18ec6d11d5fd17ca66c1153f0f4e869b6b0/SimTracker/Common/python/trackingParticleSelector_cfi.py
    // https://github.com/cms-sw/cmssw/blob/7cbdb18ec6d11d5fd17ca66c1153f0f4e869b6b0/SimTracker/Common/interface/TrackingParticleSelector.h#L122-L124
    const float vtx_z_thresh = 30;
    const float vtx_perp_thresh = 2.5;

    if (pt > PTCUT and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh){
        ana.tx.pushbackToBranch<float>(category_name + "_denom_eta", eta);
        ana.tx.pushbackToBranch<float>(category_name + "_inefficiency_denom_eta", eta);}
    if (abs(eta) < ETACUT and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh){
        ana.tx.pushbackToBranch<float>(category_name + "_denom_pt", pt);
        ana.tx.pushbackToBranch<float>(category_name + "_inefficiency_denom_pt", pt);}
    if (abs(eta) < ETACUT and pt > PTCUT and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh){
        ana.tx.pushbackToBranch<float>(category_name + "_denom_phi", phi);
        ana.tx.pushbackToBranch<float>(category_name + "_inefficiency_denom_phi", phi);}
    if (abs(eta) < ETACUT and pt > PTCUT and abs(vtx_z) < vtx_z_thresh){
        ana.tx.pushbackToBranch<float>(category_name + "_denom_dxy", dxy);
        ana.tx.pushbackToBranch<float>(category_name + "_inefficiency_denom_dxy", dxy);}
    if (abs(eta) < ETACUT and pt > PTCUT and abs(vtx_perp) < vtx_perp_thresh){
        ana.tx.pushbackToBranch<float>(category_name + "_denom_dz", dz);
        ana.tx.pushbackToBranch<float>(category_name + "_inefficiency_denom_dz", dz);}
    if (abs(eta) < ETACUT and pt > PTCUT and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh){
        ana.tx.pushbackToBranch<float>(category_name + "_denom_layers", lay);
        ana.tx.pushbackToBranch<float>(category_name + "_inefficiency_denom_layers", lay);
        ana.tx.pushbackToBranch<float>(category_name + "_denom_layersgap", laygap);
        ana.tx.pushbackToBranch<float>(category_name + "_inefficiency_denom_layersgap", laygap);
        ana.tx.pushbackToBranch<float>(category_name + "_denom_hits", hits);
        ana.tx.pushbackToBranch<float>(category_name + "_inefficiency_denom_hits", hits);
    }

    if (effset.pass(isimtrk))
    {
        if (pt > PTCUT and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh)
            ana.tx.pushbackToBranch<float>(category_name + "_numer_eta", eta);
        if (abs(eta) < ETACUT and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh)
            ana.tx.pushbackToBranch<float>(category_name + "_numer_pt", pt);
        if (abs(eta) < ETACUT and pt > PTCUT and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh)
            ana.tx.pushbackToBranch<float>(category_name + "_numer_phi", phi);
        if (abs(eta) < ETACUT and pt > PTCUT and abs(vtx_z) < vtx_z_thresh)
            ana.tx.pushbackToBranch<float>(category_name + "_numer_dxy", dxy);
        if (abs(eta) < ETACUT and pt > PTCUT and abs(vtx_perp) < vtx_perp_thresh)
            ana.tx.pushbackToBranch<float>(category_name + "_numer_dz", dz);
        //printf("layer: %f\n",lay);
        if (abs(eta) < ETACUT and pt > PTCUT and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh){
            ana.tx.pushbackToBranch<float>(category_name + "_numer_hits", hits);
            ana.tx.pushbackToBranch<float>(category_name + "_numer_layers", lay);
            ana.tx.pushbackToBranch<float>(category_name + "_numer_layersgap", laygap);}
    }
    //inefficiencies
    else
    {
        if (pt > PTCUT and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh)
            ana.tx.pushbackToBranch<float>(category_name + "_inefficiency_numer_eta", eta);
        if (abs(eta) < ETACUT and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh)
            ana.tx.pushbackToBranch<float>(category_name + "_inefficiency_numer_pt", pt);
        if (abs(eta) < ETACUT and pt > PTCUT and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh)
            ana.tx.pushbackToBranch<float>(category_name + "_inefficiency_numer_phi", phi);
        if (abs(eta) < ETACUT and pt > PTCUT and abs(vtx_z) < vtx_z_thresh)
            ana.tx.pushbackToBranch<float>(category_name + "_inefficiency_numer_dxy", dxy);
        if (abs(eta) < ETACUT and pt > PTCUT and abs(vtx_perp) < vtx_perp_thresh)
            ana.tx.pushbackToBranch<float>(category_name + "_inefficiency_numer_dz", dz);
        if (abs(eta) < ETACUT and pt > PTCUT and abs(vtx_z) < vtx_z_thresh and abs(vtx_perp) < vtx_perp_thresh){
            ana.tx.pushbackToBranch<float>(category_name + "_inefficiency_numer_hits", hits);
            ana.tx.pushbackToBranch<float>(category_name + "_inefficiency_numer_layers", lay);
            ana.tx.pushbackToBranch<float>(category_name + "_inefficiency_numer_layersgap", laygap);}
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
    ana.histograms.addVecHistogram(category_name + "_h_fakerate_denom_eta" , 180 , -4.5  , 4.5  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_fakerate_denom_eta"); } );
    ana.histograms.addVecHistogram(category_name + "_h_fakerate_numer_eta" , 180 , -4.5  , 4.5  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_fakerate_numer_eta"); } );
    ana.histograms.addVecHistogram(category_name + "_h_fakerate_denom_phi" , 180 , -M_PI , M_PI , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_fakerate_denom_phi"); } );
    ana.histograms.addVecHistogram(category_name + "_h_fakerate_numer_phi" , 180 , -M_PI , M_PI , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_fakerate_numer_phi"); } );

}

void fillFakeRateSets(std::vector<FakeRateSetDefinition>& FRsets)
{
    for (auto& FRset : FRsets)
    {
        if (FRset.set_name.Contains("TC_"))
        {
            for (unsigned int itc = 0; itc < sdl.tc_pt().size(); ++itc)
            {
                fillFakeRateSet(itc, FRset);
            }
        }
        else if (FRset.set_name.Contains("T4s_"))
        {
            for (unsigned int it4 = 0; it4 < sdl.t4_pt().size(); ++it4)
            {
                fillFakeRateSet(it4, FRset);
            }
        }
        else if (FRset.set_name.Contains("pT3_"))
        {
            for (unsigned int ipt3 = 0; ipt3 < sdl.pT3_pt().size(); ++ipt3)
            {
                fillFakeRateSet(ipt3, FRset);
            }
        }
        else if (FRset.set_name.Contains("pLS_"))
        {
            for (unsigned int ipls = 0; ipls < sdl.pLS_pt().size(); ++ipls)
            {
                fillFakeRateSet(ipls, FRset);
            }
        }
        else if (FRset.set_name.Contains("pT5_"))
        {
            for (unsigned int ipt5 = 0; ipt5 < sdl.pT5_pt().size(); ++ipt5)
            {
                fillFakeRateSet(ipt5, FRset);
            }
        }

        else if (FRset.set_name.Contains("pT4_"))
        {
            for (unsigned int ipt4 = 0; ipt4 < sdl.pT4_pt().size(); ++ipt4)
            {
                fillFakeRateSet(ipt4, FRset);
            }
        }
        else if (FRset.set_name.Contains("T3_"))
        {
            for (unsigned int it3 = 0; it3 < sdl.t3_pt().size(); ++it3)
            {
                fillFakeRateSet(it3, FRset);
            }
        }
        else if (FRset.set_name.Contains("T5_"))
        {
            for (unsigned int it5 = 0; it5 < sdl.t5_pt().size(); ++it5)
            {
                fillFakeRateSet(it5, FRset);
            }
        }
        else if(FRset.set_name.Contains("TCE_"))
        {
            for(unsigned int iTCE = 0; iTCE < sdl.tce_pt().size(); ++iTCE)
            {
                fillFakeRateSet(iTCE, FRset);
            }
        }
    }
}

void fillFakeRateSet(int itc, FakeRateSetDefinition& FRset)
{
    float pt = 0;
    float eta = 0;
    float phi = 0;
    if (FRset.set_name.Contains("TC_"))
    {
        pt = sdl.tc_pt()[itc];
        eta = sdl.tc_eta()[itc];
        phi = sdl.tc_phi()[itc];
    }
    else if (FRset.set_name.Contains("T4s_"))
    {
        pt = sdl.t4_pt()[itc];
        eta = sdl.t4_eta()[itc];
        phi = sdl.t4_phi()[itc];
    }
    else if (FRset.set_name.Contains("pT3_"))
    {
        pt = sdl.pT3_pt()[itc];
        eta = sdl.pT3_eta()[itc];
        phi = sdl.pT3_phi()[itc];
    }
    else if (FRset.set_name.Contains("pLS_"))
    {
        pt = sdl.pLS_pt()[itc];
        eta = sdl.pLS_eta()[itc];
        phi = sdl.pLS_phi()[itc];
    }
    else if (FRset.set_name.Contains("pT5_"))
    {
        pt = sdl.pT5_pt()[itc];
        eta = sdl.pT5_eta()[itc];
        phi = sdl.pT5_phi()[itc];
    }
    else if (FRset.set_name.Contains("pT4_"))
    {
        pt = sdl.pT4_pt()[itc];
        eta = sdl.pT4_eta()[itc];
        phi = sdl.pT4_phi()[itc];
    }
    else if (FRset.set_name.Contains("T3_"))
    {
        pt = sdl.t3_pt()[itc];
        eta = sdl.t3_eta()[itc];
        phi = sdl.t3_phi()[itc];
    }
    else if (FRset.set_name.Contains("T5_"))
    {
        pt = sdl.t5_pt()[itc];
        eta = sdl.t5_eta()[itc];
        phi = sdl.t5_phi()[itc];
    }
    else if (FRset.set_name.Contains("TCE_"))
    {
        pt = sdl.tce_pt()[itc];
        eta = sdl.tce_eta()[itc];
        phi = sdl.tce_phi()[itc];
    }

    TString category_name = FRset.set_name;

    if (pt > PTCUT)
        ana.tx.pushbackToBranch<float>(category_name + "_fakerate_denom_eta", eta);
    if (abs(eta) < ETACUT)
        ana.tx.pushbackToBranch<float>(category_name + "_fakerate_denom_pt", pt);
    if (abs(eta) < ETACUT and pt > PTCUT)
        ana.tx.pushbackToBranch<float>(category_name + "_fakerate_denom_phi", phi);

    if (FRset.pass(itc))
    {
        if (pt > PTCUT)
            ana.tx.pushbackToBranch<float>(category_name + "_fakerate_numer_eta", eta);
        if (abs(eta) < ETACUT)
            ana.tx.pushbackToBranch<float>(category_name + "_fakerate_numer_pt", pt);
        if (abs(eta) < ETACUT and pt > PTCUT)
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
    ana.histograms.addVecHistogram(category_name + "_h_duplrate_denom_eta" , 180 , -4.5  , 4.5  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_duplrate_denom_eta"); } );
    ana.histograms.addVecHistogram(category_name + "_h_duplrate_numer_eta" , 180 , -4.5  , 4.5  , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_duplrate_numer_eta"); } );
    ana.histograms.addVecHistogram(category_name + "_h_duplrate_denom_phi" , 180 , -M_PI , M_PI , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_duplrate_denom_phi"); } );
    ana.histograms.addVecHistogram(category_name + "_h_duplrate_numer_phi" , 180 , -M_PI , M_PI , [&, category_name]() { return ana.tx.getBranchLazy<vector<float>>(category_name + "_duplrate_numer_phi"); } );

}

void fillDuplicateRateSets(std::vector<DuplicateRateSetDefinition>& DLsets)
{
    for (auto& DLset : DLsets)
    {
        if (DLset.set_name.Contains("TC_"))
        {
            for (unsigned int itc = 0; itc < sdl.tc_pt().size(); ++itc)
            {
                fillDuplicateRateSet(itc, DLset);
            }
        }
        else if (DLset.set_name.Contains("T4s_"))
        {
            for (unsigned int it4 = 0; it4 < sdl.t4_pt().size(); ++it4)
            {
                fillDuplicateRateSet(it4, DLset);
            }
        }
        else if (DLset.set_name.Contains("pT3_"))
        {
            for (unsigned int ipt3 = 0; ipt3 < sdl.pT3_pt().size(); ++ipt3)
            {
                fillDuplicateRateSet(ipt3, DLset);
            }
        }
        else if (DLset.set_name.Contains("pLS_"))
        {
            for (unsigned int ipls = 0; ipls < sdl.pLS_pt().size(); ++ipls)
            {
                fillDuplicateRateSet(ipls, DLset);
            }
        }
        else if (DLset.set_name.Contains("pT5_"))
        {
            for (unsigned int ipt5 = 0; ipt5 < sdl.pT5_pt().size(); ++ipt5)
            {
                fillDuplicateRateSet(ipt5, DLset);
            }
        }
        else if (DLset.set_name.Contains("pT4_"))
        {
            for (unsigned int ipt4 = 0; ipt4 < sdl.pT4_pt().size(); ++ipt4)
            {
                fillDuplicateRateSet(ipt4, DLset);
            }
        }
        else if (DLset.set_name.Contains("T3_"))
        {
            for (unsigned int it3 = 0; it3 < sdl.t3_pt().size(); ++it3)
            {
                fillDuplicateRateSet(it3, DLset);
            }
        }
        else if (DLset.set_name.Contains("T5_"))
        {
            for (unsigned int it5 = 0; it5 < sdl.t5_pt().size(); ++it5)
            {
                fillDuplicateRateSet(it5, DLset);
            }
        }
        else if(DLset.set_name.Contains("TCE_"))
        {
            for(unsigned int itce = 0; itce < sdl.tce_pt().size(); ++itce)
            {
                fillDuplicateRateSet(itce, DLset);
            }
        }
    }
}

void fillDuplicateRateSet(int itc, DuplicateRateSetDefinition& DLset)
{
    float pt = 0;
    float eta = 0;
    float phi = 0;
    if (DLset.set_name.Contains("TC_"))
    {
        pt = sdl.tc_pt()[itc];
        eta = sdl.tc_eta()[itc];
        phi = sdl.tc_phi()[itc];
    }
    else if(DLset.set_name.Contains("TCE_"))
    {
        pt = sdl.tce_pt()[itc];
        eta = sdl.tce_eta()[itc];
        phi = sdl.tce_phi()[itc];
    }
    else if (DLset.set_name.Contains("T4s_"))
    {
        pt = sdl.t4_pt()[itc];
        eta = sdl.t4_eta()[itc];
        phi = sdl.t4_phi()[itc];
    }
    else if (DLset.set_name.Contains("pT3_"))
    {
        pt = sdl.pT3_pt()[itc];
        eta = sdl.pT3_eta()[itc];
        phi = sdl.pT3_phi()[itc];
    }
    else if (DLset.set_name.Contains("pLS_"))
    {
        pt = sdl.pLS_pt()[itc];
        eta = sdl.pLS_eta()[itc];
        phi = sdl.pLS_phi()[itc];
    }
    else if (DLset.set_name.Contains("pT5_"))
    {
        pt = sdl.pT5_pt()[itc];
        eta = sdl.pT5_eta()[itc];
        phi = sdl.pT5_phi()[itc];
    }
    else if (DLset.set_name.Contains("pT4_"))
    {
        pt = sdl.pT4_pt()[itc];
        eta = sdl.pT4_eta()[itc];
        phi = sdl.pT4_phi()[itc];
    }
    else if (DLset.set_name.Contains("T3_"))
    {
        pt = sdl.t3_pt()[itc];
        eta = sdl.t3_eta()[itc];
        phi = sdl.t3_phi()[itc];
    }
    else if (DLset.set_name.Contains("T5_"))
    {
        pt = sdl.t5_pt()[itc];
        eta = sdl.t5_eta()[itc];
        phi = sdl.t5_phi()[itc];
    }

    TString category_name = DLset.set_name;

    if (pt > PTCUT)
        ana.tx.pushbackToBranch<float>(category_name + "_duplrate_denom_eta", eta);
    if (abs(eta) < ETACUT)
        ana.tx.pushbackToBranch<float>(category_name + "_duplrate_denom_pt", pt);
    if (abs(eta) < ETACUT and pt > PTCUT)
        ana.tx.pushbackToBranch<float>(category_name + "_duplrate_denom_phi", phi);

    if (DLset.pass(itc))
    {
        if (pt > PTCUT)
            ana.tx.pushbackToBranch<float>(category_name + "_duplrate_numer_eta", eta);
        if (abs(eta) < ETACUT)
            ana.tx.pushbackToBranch<float>(category_name + "_duplrate_numer_pt", pt);
        if (abs(eta) < ETACUT and pt > PTCUT)
            ana.tx.pushbackToBranch<float>(category_name + "_duplrate_numer_phi", phi);
    }
}

