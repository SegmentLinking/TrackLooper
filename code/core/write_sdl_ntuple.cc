#include "write_sdl_ntuple.h"

#define N_MAX_MD_PER_MODULES 89
#define N_MAX_SEGMENTS_PER_MODULE 537
#define MAX_NTRIPLET_PER_MODULE 1170
#define MAX_NQUINTUPLET_PER_MODULE 513


//________________________________________________________________________________________________________________________________
void createOutputBranches()
{
    createOutputBranches_v2();
}

//________________________________________________________________________________________________________________________________
void fillOutputBranches(SDL::Event* event)
{
    fillOutputBranches_v2(event);
}

//________________________________________________________________________________________________________________________________
void createOutputBranches_v2()
{
    // Setup output TTree
    ana.tx->createBranch<vector<float>>("sim_pt");
    ana.tx->createBranch<vector<float>>("sim_eta");
    ana.tx->createBranch<vector<float>>("sim_phi");
    ana.tx->createBranch<vector<float>>("sim_pca_dxy");
    ana.tx->createBranch<vector<float>>("sim_pca_dz");
    ana.tx->createBranch<vector<int>>("sim_q");
    ana.tx->createBranch<vector<int>>("sim_event");
    ana.tx->createBranch<vector<int>>("sim_pdgId");
    ana.tx->createBranch<vector<float>>("sim_vx");
    ana.tx->createBranch<vector<float>>("sim_vy");
    ana.tx->createBranch<vector<float>>("sim_vz");
    ana.tx->createBranch<vector<float>>("sim_trkNtupIdx");
    ana.tx->createBranch<vector<int>>("sim_TC_matched");
    ana.tx->createBranch<vector<int>>("sim_TC_matched_mask");

    // Track candidates
    ana.tx->createBranch<vector<float>>("tc_pt");
    ana.tx->createBranch<vector<float>>("tc_eta");
    ana.tx->createBranch<vector<float>>("tc_phi");
    ana.tx->createBranch<vector<int>>("tc_type");
    ana.tx->createBranch<vector<int>>("tc_isFake");
    ana.tx->createBranch<vector<int>>("tc_isDuplicate");
    ana.tx->createBranch<vector<vector<int>>>("tc_matched_simIdx");
}

//________________________________________________________________________________________________________________________________
void fillOutputBranches_v2(SDL::Event* event)
{

    // ============ Sim tracks =============
    int n_accepted_simtrk = 0;
    for (unsigned int isimtrk = 0; isimtrk < trk.sim_pt().size(); ++isimtrk)
    {
        // Skip out-of-time pileup
        if (trk.sim_bunchCrossing()[isimtrk] != 0)
            continue;

        // Skip non-hard-scatter
        if (trk.sim_event()[isimtrk] != 0)
            continue;

        ana.tx->pushbackToBranch<float>("sim_pt", trk.sim_pt()[isimtrk]);
        ana.tx->pushbackToBranch<float>("sim_eta", trk.sim_eta()[isimtrk]);
        ana.tx->pushbackToBranch<float>("sim_phi", trk.sim_phi()[isimtrk]);
        ana.tx->pushbackToBranch<float>("sim_pca_dxy", trk.sim_pca_dxy()[isimtrk]);
        ana.tx->pushbackToBranch<float>("sim_pca_dz", trk.sim_pca_dz()[isimtrk]);
        ana.tx->pushbackToBranch<int>("sim_q", trk.sim_q()[isimtrk]);
        ana.tx->pushbackToBranch<int>("sim_event", trk.sim_event()[isimtrk]);
        ana.tx->pushbackToBranch<int>("sim_pdgId", trk.sim_pdgId()[isimtrk]);

        // For vertex we need to look it up from simvtx info
        int vtxidx = trk.sim_parentVtxIdx()[isimtrk];
        ana.tx->pushbackToBranch<float>("sim_vx", trk.simvtx_x()[vtxidx]);
        ana.tx->pushbackToBranch<float>("sim_vy", trk.simvtx_y()[vtxidx]);
        ana.tx->pushbackToBranch<float>("sim_vz", trk.simvtx_z()[vtxidx]);

        // The trkNtupIdx is the idx in the trackingNtuple
        ana.tx->pushbackToBranch<float>("sim_trkNtupIdx", isimtrk);

        // Increase the counter for accepted simtrk
        n_accepted_simtrk++;
    }

    // Intermediate variables to keep track of matched track candidates for a given sim track
    std::vector<int> sim_TC_matched(n_accepted_simtrk);
    std::vector<int> sim_TC_matched_mask(n_accepted_simtrk);

    // Intermediate variables to keep track of matched sim tracks for a given track candidate
    std::vector<std::vector<int>> tc_matched_simIdx;

    // ============ Track candidates =============
    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());
    unsigned int nTrackCandidates = *trackCandidatesInGPU.nTrackCandidates;
    for (unsigned int idx = 0; idx < nTrackCandidates; idx++)
    {
        // Compute reco quantities of track candidate based on final object
        int type, isFake;
        float pt, eta, phi;
        std::vector<int> simidx;
        std::tie(type, pt, eta, phi, isFake, simidx) = parseTrackCandidate(event, idx);
        ana.tx->pushbackToBranch<float>("tc_pt", pt);
        ana.tx->pushbackToBranch<float>("tc_eta", eta);
        ana.tx->pushbackToBranch<float>("tc_phi", phi);
        ana.tx->pushbackToBranch<int>("tc_type", type);
        ana.tx->pushbackToBranch<int>("tc_isFake", isFake);
        tc_matched_simIdx.push_back(simidx);

        // Loop over matched sim idx and increase counter of TC_matched
        for (auto& idx : simidx)
        {
            // NOTE Important to note that the idx of the std::vector<> is same
            // as the tracking-ntuple's sim track idx ONLY because event==0 and bunchCrossing==0 condition is applied!!
            // Also do not try to access beyond the event and bunchCrossing
            if (idx < n_accepted_simtrk)
            {
                sim_TC_matched.at(idx) += 1;
                sim_TC_matched_mask.at(idx) |= (1 << type);
            }
        }
    }

    // Using the intermedaite variables to compute whether a given track candidate is a duplicate
    vector<int> tc_isDuplicate(tc_matched_simIdx.size());
    // Loop over the track candidates
    for (unsigned int i = 0; i < tc_matched_simIdx.size(); ++i)
    {
        bool isDuplicate = false;
        // Loop over the sim idx matched to this track candidate
        for (unsigned int isim = 0; isim < tc_matched_simIdx[i].size(); ++isim)
        {
            // Using the sim_TC_matched to see whether this track candidate is matched to a sim track that is matched to more than one
            int simidx = tc_matched_simIdx[i][isim];
            if (simidx < n_accepted_simtrk)
            {
                if (sim_TC_matched[simidx] > 1)
                {
                    isDuplicate = true;
                }
            }
        }
        tc_isDuplicate[i] = isDuplicate;
    }

    // Now set the last remaining branches
    ana.tx->setBranch<vector<int>>("sim_TC_matched", sim_TC_matched);
    ana.tx->setBranch<vector<int>>("sim_TC_matched_mask", sim_TC_matched_mask);
    ana.tx->setBranch<vector<vector<int>>>("tc_matched_simIdx", tc_matched_simIdx);
    ana.tx->setBranch<vector<int>>("tc_isDuplicate", tc_isDuplicate);

    ana.tx->fill();
    ana.tx->clear();
}

//________________________________________________________________________________________________________________________________
std::tuple<int, float, float, float, int, vector<int>> parseTrackCandidate(SDL::Event* event, unsigned int idx)
{
    // Get the type of the track candidate
    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());
    short type = trackCandidatesInGPU.trackCandidateType[idx];

    enum
    {
        pT5 = 7,
        pT3 = 5,
        T5 = 4,
        pLS = 8
    };

    // Compute pt eta phi and hit indices that will be used to figure out whether the TC matched
    float pt, eta, phi;
    std::vector<unsigned int> hit_idx, hit_type;
    switch (type)
    {
        case pT5: std::tie(pt, eta, phi, hit_idx, hit_type) = parsepT5(event, idx); break;
        case pT3: std::tie(pt, eta, phi, hit_idx, hit_type) = parsepT3(event, idx); break;
        case T5:  std::tie(pt, eta, phi, hit_idx, hit_type) = parseT5(event, idx); break;
        case pLS: std::tie(pt, eta, phi, hit_idx, hit_type) = parsepLS(event, idx); break;

    }

    // Perform matching
    std::vector<int> simidx = matchedSimTrkIdxs(hit_idx, hit_type);
    int isFake = simidx.size() == 0;

    return {type, pt, eta, phi, isFake, simidx};
}

//________________________________________________________________________________________________________________________________
std::tuple<float, float, float, vector<unsigned int>, vector<unsigned int>> parsepT5(SDL::Event* event, unsigned int idx)
{
    // Get relevant information
    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());
    SDL::triplets& tripletsInGPU = (*event->getTriplets());
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());

    //
    // pictorial representation of a pT5
    //
    // inner tracker        outer tracker
    // -------------  --------------------------
    // pLS            01    23    45    67    89   (anchor hit of a minidoublet is always the first of the pair)
    // ****           oo -- oo -- oo -- oo -- oo   pT5
    //                oo -- oo -- oo               first T3 of the T5
    //                            oo -- oo -- oo   second T3 of the T5
    unsigned int pT5 = trackCandidatesInGPU.directObjectIndices[idx];
    std::vector<unsigned int> Hits = getHitsFrompT5(event, pT5);
    unsigned int Hit_0 = Hits[0];
    unsigned int Hit_4 = Hits[4];
    unsigned int Hit_8 = Hits[8];

    std::vector<unsigned int> T3s = getT3sFrompT5(event, pT5);
    unsigned int T3_0 = T3s[0];
    unsigned int T3_1 = T3s[1];

    unsigned int pLS = getPixelLSFrompT5(event, pT5);

    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;

    //=================================================================================
    // Some history and geometry lesson...
    // For a given T3, we compute two angles. (NOTE: This is a bit weird!)
    // Historically, T3 were created out of T4, which we used to build a long time ago.
    // So for the sake of argument let's discuss T4 first.
    // For a T4, we have 4 mini-doublets.
    // Therefore we have 4 "anchor hits".
    // Therefore we have 4 xyz points.
    //
    //
    //       *
    //       |\
    //       | \
    //       |1 \
    //       |   \
    //       |  * \
    //       |
    //       |
    //       |
    //       |
    //       |
    //       |  * /
    //       |   /
    //       |2 /
    //       | /
    //       |/
    //       *
    //
    //
    // Then from these 4 points, one can approximate a some sort of "best" fitted circle trajectory,
    // and obtain "tangential" angles from 1st and 4th hits.
    // See the carton below.
    // The "*" are the 4 physical hit points
    // angle 1 and 2 are the "tangential" angle for a "circle" from 4 * points.
    // Please note, that a straight line from first two * and the latter two * are NOT the
    // angle 1 and angle 2. (they were called "beta" angles)
    // But rather, a slightly larger angle.
    // Because 4 * points would be on a circle, and a tangential line on the circles
    // would deviate from the points on circles.
    //
    // In the early days of LST, there was an iterative algorithm (devised by Slava) to
    // obtain the angle beta1 and 2 _without_ actually performing a 4 point circle fit.
    // Hence, the beta1 and beta2 were quickly estimated without too many math operations
    // and afterwards (beta1-beta2) was computed to obtain what we call a "delta-beta" values.
    //
    // For a real track, the deltabeta ~ 0, for fakes, it'd have a flat distribution.
    //
    // However, after some time we abandonded the T4s, and moved to T3s.
    // In T3, however, now we have the following cartoon:
    //
    //       *
    //       |\
    //       | \
    //       |1 \
    //       |   \
    //       |  * X   (* here are "two" MDs but really just one)
    //       |   /
    //       |2 /
    //       | /
    //       |/
    //       *
    //
    // With the "four" *'s (really just "three") you can still run the iterative beta calculation,
    // which is what we still currently do, we still get two beta1 and beta2
    // But! high school geometry tells us that 3 points = ONLY 1 possible CIRCLE!
    // There is really nothing to "fit" here.
    // YET we still compute these in T3, out of legacy method of how we used to treat T4s.
    //
    // Hence, in the below code, "betaIn_in" and "betaOut_in" if we performed
    // a circle fit they would come out by definition identical values.
    // But due to our approximate iterative beta calculation method, they come out different values.
    // So if we are "cutting on" abs(deltaBeta) = abs(betaIn_in - betaOut_in) < threshold,
    // what does that even mean?
    //
    // Anyhow, as of now, we compute 2 beta's for T3s, and T5 has two T3s.
    // And from there we estimate the pt's and we compute pt_T5.

    // Compute the radial distance between first mini-doublet to third minidoublet
    const float dr_in = sqrt(pow(hitsInGPU.xs[Hit_4] - hitsInGPU.xs[Hit_0], 2) + pow(hitsInGPU.ys[Hit_4] - hitsInGPU.ys[Hit_0], 2));
    // Compute the radial distance between third mini-doublet to fifth minidoublet
    const float dr_out = sqrt(pow(hitsInGPU.xs[Hit_8] - hitsInGPU.xs[Hit_4], 2) + pow(hitsInGPU.ys[Hit_8] - hitsInGPU.ys[Hit_4], 2));
    float betaIn_in   = __H2F(tripletsInGPU.betaIn[T3_0]);
    float betaOut_in  = __H2F(tripletsInGPU.betaOut[T3_0]);
    float betaIn_out  = __H2F(tripletsInGPU.betaIn[T3_1]);
    float betaOut_out = __H2F(tripletsInGPU.betaOut[T3_1]);
    const float ptAv_in = abs(dr_in * k2Rinv1GeVf / sin((betaIn_in + betaOut_in) / 2.));
    const float ptAv_out = abs(dr_out * k2Rinv1GeVf / sin((betaIn_out + betaOut_out) / 2.));
    const float pt_T5 = (ptAv_in + ptAv_out) / 2.;

    // pixel pt
    const float pt_pLS = segmentsInGPU.ptIn[pLS];
    const float eta_pLS = segmentsInGPU.eta[pLS];
    const float phi_pLS = segmentsInGPU.phi[pLS];

    // average pt
    const float pt = (pt_pLS + pt_T5) / 2.;

    // Form the hit idx/type vector
    std::vector<unsigned int> hit_idx = getHitIdxsFrompT5(event, pT5);
    std::vector<unsigned int> hit_type = getHitTypesFrompT5(event, pT5);

    return {pt, eta_pLS, phi_pLS, hit_idx, hit_type};

}

//________________________________________________________________________________________________________________________________
std::tuple<float, float, float, vector<unsigned int>, vector<unsigned int>> parsepT3(SDL::Event* event, unsigned int idx)
{
    // Get relevant information
    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());
    SDL::triplets& tripletsInGPU = (*event->getTriplets());
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());

    //
    // pictorial representation of a pT3
    //
    // inner tracker        outer tracker
    // -------------  --------------------------
    // pLS            01    23    45               (anchor hit of a minidoublet is always the first of the pair)
    // ****           oo -- oo -- oo               pT3
    unsigned int pT3 = trackCandidatesInGPU.directObjectIndices[idx];
    std::vector<unsigned int> Hits = getHitsFrompT3(event, pT3);
    unsigned int Hit_0 = Hits[0];
    unsigned int Hit_4 = Hits[4];

    unsigned int T3 = getT3FrompT3(event, pT3);

    unsigned int pLS = getPixelLSFrompT3(event, pT3);

    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
    const float dr = sqrt(pow(hitsInGPU.xs[Hit_4] - hitsInGPU.xs[Hit_0], 2) + pow(hitsInGPU.ys[Hit_4] - hitsInGPU.ys[Hit_0], 2));
    float betaIn   = __H2F(tripletsInGPU.betaIn[T3]);
    float betaOut  = __H2F(tripletsInGPU.betaOut[T3]);
    const float pt_T3 = abs(dr * k2Rinv1GeVf / sin((betaIn + betaOut) / 2.));

    // pixel pt
    const float pt_pLS = segmentsInGPU.ptIn[pLS];
    const float eta_pLS = segmentsInGPU.eta[pLS];
    const float phi_pLS = segmentsInGPU.phi[pLS];

    // average pt
    const float pt = (pt_pLS + pt_T3) / 2.;

    // Form the hit idx/type vector
    std::vector<unsigned int> hit_idx = getHitIdxsFrompT3(event, pT3);
    std::vector<unsigned int> hit_type = getHitTypesFrompT3(event, pT3);

    return {pt, eta_pLS, phi_pLS, hit_idx, hit_type};

}

//________________________________________________________________________________________________________________________________
std::tuple<float, float, float, vector<unsigned int>, vector<unsigned int>> parseT5(SDL::Event* event, unsigned int idx)
{
    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());
    SDL::triplets& tripletsInGPU = (*event->getTriplets());
    SDL::hits& hitsInGPU = (*event->getHits());
    unsigned int T5 = trackCandidatesInGPU.directObjectIndices[idx];
    std::vector<unsigned int> T3s = getT3sFromT5(event, T5);
    std::vector<unsigned int> hits = getHitsFromT5(event, T5);

    //
    // pictorial representation of a T5
    //
    // inner tracker        outer tracker
    // -------------  --------------------------
    //                01    23    45    67    89   (anchor hit of a minidoublet is always the first of the pair)
    //  (none)        oo -- oo -- oo -- oo -- oo   T5
    unsigned int Hit_0 = hits[0];
    unsigned int Hit_4 = hits[4];
    unsigned int Hit_8 = hits[8];

    // radial distance
    const float dr_in = sqrt(pow(hitsInGPU.xs[Hit_4] - hitsInGPU.xs[Hit_0], 2) + pow(hitsInGPU.ys[Hit_4] - hitsInGPU.ys[Hit_0], 2));
    const float dr_out = sqrt(pow(hitsInGPU.xs[Hit_8] - hitsInGPU.xs[Hit_4], 2) + pow(hitsInGPU.ys[Hit_8] - hitsInGPU.ys[Hit_4], 2));

    // beta angles
    float betaIn_in   = __H2F(tripletsInGPU.betaIn [T3s[0]]);
    float betaOut_in  = __H2F(tripletsInGPU.betaOut[T3s[0]]);
    float betaIn_out  = __H2F(tripletsInGPU.betaIn [T3s[1]]);
    float betaOut_out = __H2F(tripletsInGPU.betaOut[T3s[1]]);

    // constants
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;

    // Compute pt estimates from inner and outer triplets
    const float ptAv_in = abs(dr_in * k2Rinv1GeVf / sin((betaIn_in + betaOut_in) / 2.));
    const float ptAv_out = abs(dr_out * k2Rinv1GeVf / sin((betaIn_out + betaOut_out) / 2.));

    // T5 pt is average of the two pt estimates
    const float pt = (ptAv_in + ptAv_out) / 2.;

    // T5 eta and phi are computed using outer and innermost hits
    SDL::CPU::Hit hitA(trk.ph2_x()[Hit_0], trk.ph2_y()[Hit_0], trk.ph2_z()[Hit_0]);
    SDL::CPU::Hit hitB(trk.ph2_x()[Hit_8], trk.ph2_y()[Hit_8], trk.ph2_z()[Hit_8]);
    const float phi = hitA.phi();
    const float eta = hitB.eta();

    std::vector<unsigned int> hit_idx = getHitIdxsFromT5(event, T5);
    std::vector<unsigned int> hit_type = getHitTypesFromT5(event, T5);

    return {pt, eta, phi, hit_idx, hit_type};
}

//________________________________________________________________________________________________________________________________
std::tuple<float, float, float, vector<unsigned int>, vector<unsigned int>> parsepLS(SDL::Event* event, unsigned int idx)
{
    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());
    SDL::segments& segmentsInGPU = (*event->getSegments());

    // Getting pLS index
    unsigned int pLS = trackCandidatesInGPU.directObjectIndices[idx];

    // Getting pt eta and phi
    float pt = segmentsInGPU.ptIn[pLS];
    float eta = segmentsInGPU.eta[pLS];
    float phi = segmentsInGPU.phi[pLS];

    // Getting hit indices and types
    std::vector<unsigned int> hit_idx = getPixelHitIdxsFrompLS(event, pLS);
    std::vector<unsigned int> hit_type = getPixelHitTypesFrompLS(event, pLS);

    return {pt, eta, phi, hit_idx, hit_type};
}

//________________________________________________________________________________________________________________________________
float computeRadiusFromThreeAnchorHits(float x1, float y1, float x2, float y2, float x3, float y3, float& g, float& f)
{
   float radius = 0;
   if ((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3) == 0)
    {
        return -1; // WTF man three collinear points!
    }

    float denom = ((y1 - y3) * (x2 - x3) - (x1 - x3) * (y2 - y3));

    g = 0.5 * ((y3 - y2) * (x1 * x1 + y1 * y1) + (y1 - y3) * (x2 * x2 + y2 * y2) + (y2 - y1) * (x3 * x3 + y3 * y3)) / denom;

    f = 0.5 * ((x2 - x3) * (x1 * x1 + y1 * y1) + (x3 - x1) * (x2 * x2 + y2 * y2) + (x1 - x2) * (x3 * x3 + y3 * y3)) / denom;

    float c = ((x2 * y3 - x3 * y2) * (x1 * x1 + y1 * y1) + (x3 * y1 - x1 * y3) * (x2 * x2 + y2 * y2) + (x1 * y2 - x2 * y1) * (x3 * x3 + y3 * y3)) / denom;

    if (g * g + f * f - c < 0)
    {
        std::cout << "FATAL! r^2 < 0!" << std::endl;
        return -1;
    }

    radius = sqrtf(g * g + f * f - c);
    return radius;
}

//________________________________________________________________________________________________________________________________
void printHitMultiplicities(SDL::Event* event)
{
    //SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());
    SDL::objectRanges& rangesInGPU = (*event->getRanges());

    int nHits = 0;
    for (unsigned int idx = 0; idx <= *(modulesInGPU.nLowerModules); idx++) // "<=" because cheating to include pixel track candidate lower module
    {
        nHits += rangesInGPU.hitRanges[4 * idx + 1] - rangesInGPU.hitRanges[4 * idx] + 1;
        nHits += rangesInGPU.hitRanges[4 * idx + 3] - rangesInGPU.hitRanges[4 * idx + 2] + 1;
    }
    std::cout <<  " nHits: " << nHits <<  std::endl;
}

//________________________________________________________________________________________________________________________________
void printMiniDoubletMultiplicities(SDL::Event* event)
{
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::modules& modulesInGPU = (*event->getModules());

    int nMiniDoublets = 0;
    int totOccupancyMiniDoublets = 0;
    for (unsigned int idx = 0; idx <= *(modulesInGPU.nModules); idx++) // "<=" because cheating to include pixel track candidate lower module
    {
        if(modulesInGPU.isLower[idx])
        {
            nMiniDoublets += miniDoubletsInGPU.nMDs[idx];
            totOccupancyMiniDoublets += miniDoubletsInGPU.totOccupancyMDs[idx];
        }
    }
    std::cout <<  " nMiniDoublets: " << nMiniDoublets <<  std::endl;
    std::cout <<  " totOccupancyMiniDoublets (including trucated ones): " << totOccupancyMiniDoublets <<  std::endl;
}

//________________________________________________________________________________________________________________________________
void printAllObjects(SDL::Event* event)
{
    printMDs(event);
    printLSs(event);
    printpLSs(event);
    printT3s(event);
}

//________________________________________________________________________________________________________________________________
void printMDs(SDL::Event* event)
{
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());
    for (unsigned int idx = 0; idx <= *(modulesInGPU.nLowerModules); ++idx)
    {
        for (unsigned int jdx = 0; jdx < miniDoubletsInGPU.nMDs[2*idx]; jdx++)
        {
            unsigned int mdIdx = (2*idx) * N_MAX_MD_PER_MODULES + jdx;
            unsigned int LowerHitIndex = miniDoubletsInGPU.anchorHitIndices[mdIdx];
            unsigned int UpperHitIndex = miniDoubletsInGPU.outerHitIndices[mdIdx];
            unsigned int hit0 = hitsInGPU.idxs[LowerHitIndex];
            unsigned int hit1 = hitsInGPU.idxs[UpperHitIndex];
            std::cout <<  "VALIDATION 'MD': " << "MD" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  std::endl;
        }
        for (unsigned int jdx = 0; jdx < miniDoubletsInGPU.nMDs[2*idx+1]; jdx++)
        {
            unsigned int mdIdx = (2*idx+1) * N_MAX_MD_PER_MODULES + jdx;
            unsigned int LowerHitIndex = miniDoubletsInGPU.anchorHitIndices[mdIdx];
            unsigned int UpperHitIndex = miniDoubletsInGPU.outerHitIndices[mdIdx];
            unsigned int hit0 = hitsInGPU.idxs[LowerHitIndex];
            unsigned int hit1 = hitsInGPU.idxs[UpperHitIndex];
            std::cout <<  "VALIDATION 'MD': " << "MD" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  std::endl;
        }
    }
}

//________________________________________________________________________________________________________________________________
void printLSs(SDL::Event* event)
{
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());
    SDL::objectRanges& rangesInGPU = (*event->getRanges());

    int nSegments = 0;
    for (unsigned int i = 0; i <  *(modulesInGPU.nLowerModules); ++i)
    {
        unsigned int idx = i;//modulesInGPU.lowerModuleIndices[i];
        nSegments += segmentsInGPU.nSegments[idx];
        for (unsigned int jdx = 0; jdx < segmentsInGPU.nSegments[idx]; jdx++)
        {
            unsigned int sgIdx = rangesInGPU.segmentModuleIndices[idx] + jdx;
            unsigned int InnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * sgIdx];
            unsigned int OuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * sgIdx + 1];
            unsigned int InnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[InnerMiniDoubletIndex];
            unsigned int InnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[InnerMiniDoubletIndex];
            unsigned int OuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[OuterMiniDoubletIndex];
            unsigned int OuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[OuterMiniDoubletIndex];
            unsigned int hit0 = hitsInGPU.idxs[InnerMiniDoubletLowerHitIndex];
            unsigned int hit1 = hitsInGPU.idxs[InnerMiniDoubletUpperHitIndex];
            unsigned int hit2 = hitsInGPU.idxs[OuterMiniDoubletLowerHitIndex];
            unsigned int hit3 = hitsInGPU.idxs[OuterMiniDoubletUpperHitIndex];
            std::cout <<  "VALIDATION 'LS': " << "LS" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  std::endl;
        }
    }
    std::cout <<  "VALIDATION nSegments: " << nSegments <<  std::endl;
}

//________________________________________________________________________________________________________________________________
void printpLSs(SDL::Event* event)
{
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());
    SDL::objectRanges& rangesInGPU = (*event->getRanges());

    unsigned int i = *(modulesInGPU.nLowerModules);
    unsigned int idx = i;//modulesInGPU.lowerModuleIndices[i];
    int npLS = segmentsInGPU.nSegments[idx];
    for (unsigned int jdx = 0; jdx < segmentsInGPU.nSegments[idx]; jdx++)
    {
        unsigned int sgIdx = rangesInGPU.segmentModuleIndices[idx] + jdx;
        unsigned int InnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * sgIdx];
        unsigned int OuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * sgIdx + 1];
        unsigned int InnerMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[InnerMiniDoubletIndex];
        unsigned int InnerMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[InnerMiniDoubletIndex];
        unsigned int OuterMiniDoubletLowerHitIndex = miniDoubletsInGPU.anchorHitIndices[OuterMiniDoubletIndex];
        unsigned int OuterMiniDoubletUpperHitIndex = miniDoubletsInGPU.outerHitIndices[OuterMiniDoubletIndex];
        unsigned int hit0 = hitsInGPU.idxs[InnerMiniDoubletLowerHitIndex];
        unsigned int hit1 = hitsInGPU.idxs[InnerMiniDoubletUpperHitIndex];
        unsigned int hit2 = hitsInGPU.idxs[OuterMiniDoubletLowerHitIndex];
        unsigned int hit3 = hitsInGPU.idxs[OuterMiniDoubletUpperHitIndex];
        std::cout <<  "VALIDATION 'pLS': " << "pLS" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  std::endl;
    }
    std::cout <<  "VALIDATION npLS: " << npLS <<  std::endl;
}

//________________________________________________________________________________________________________________________________
void printT3s(SDL::Event* event)
{
    SDL::triplets& tripletsInGPU = (*event->getTriplets());
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());
    int nTriplets = 0;
    for (unsigned int i = 0; i <  *(modulesInGPU.nLowerModules); ++i)
    {
        // unsigned int idx = SDL::modulesInGPU->lowerModuleIndices[i];
        nTriplets += tripletsInGPU.nTriplets[i];
        unsigned int idx = i;
        for (unsigned int jdx = 0; jdx < tripletsInGPU.nTriplets[idx]; jdx++)
        {
            unsigned int tpIdx = idx * 5000 + jdx;
            unsigned int InnerSegmentIndex = tripletsInGPU.segmentIndices[2 * tpIdx];
            unsigned int OuterSegmentIndex = tripletsInGPU.segmentIndices[2 * tpIdx + 1];
            unsigned int InnerSegmentInnerMiniDoubletIndex = segmentsInGPU.mdIndices[2 * InnerSegmentIndex];
            unsigned int InnerSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * InnerSegmentIndex + 1];
            unsigned int OuterSegmentOuterMiniDoubletIndex = segmentsInGPU.mdIndices[2 * OuterSegmentIndex + 1];

            unsigned int hit_idx0 = miniDoubletsInGPU.anchorHitIndices[InnerSegmentInnerMiniDoubletIndex];
            unsigned int hit_idx1 = miniDoubletsInGPU.outerHitIndices[InnerSegmentInnerMiniDoubletIndex];
            unsigned int hit_idx2 = miniDoubletsInGPU.anchorHitIndices[InnerSegmentOuterMiniDoubletIndex];
            unsigned int hit_idx3 = miniDoubletsInGPU.outerHitIndices[InnerSegmentOuterMiniDoubletIndex];
            unsigned int hit_idx4 = miniDoubletsInGPU.anchorHitIndices[OuterSegmentOuterMiniDoubletIndex];
            unsigned int hit_idx5 = miniDoubletsInGPU.outerHitIndices[OuterSegmentOuterMiniDoubletIndex];

            unsigned int hit0 = hitsInGPU.idxs[hit_idx0];
            unsigned int hit1 = hitsInGPU.idxs[hit_idx1];
            unsigned int hit2 = hitsInGPU.idxs[hit_idx2];
            unsigned int hit3 = hitsInGPU.idxs[hit_idx3];
            unsigned int hit4 = hitsInGPU.idxs[hit_idx4];
            unsigned int hit5 = hitsInGPU.idxs[hit_idx5];
            std::cout <<  "VALIDATION 'T3': " << "T3" <<  " hit0: " << hit0 <<  " hit1: " << hit1 <<  " hit2: " << hit2 <<  " hit3: " << hit3 <<  " hit4: " << hit4 <<  " hit5: " << hit5 <<  std::endl;
        }
    }
    std::cout <<  "VALIDATION nTriplets: " << nTriplets <<  std::endl;
}

//________________________________________________________________________________________________________________________________
void debugPrintOutlierMultiplicities(SDL::Event* event)
{
    SDL::trackCandidates& trackCandidatesInGPU = (*event->getTrackCandidates());
    SDL::triplets& tripletsInGPU = (*event->getTriplets());
    SDL::segments& segmentsInGPU = (*event->getSegments());
    SDL::miniDoublets& miniDoubletsInGPU = (*event->getMiniDoublets());
    //SDL::hits& hitsInGPU = (*event->getHits());
    SDL::modules& modulesInGPU = (*event->getModules());
    SDL::objectRanges& rangesInGPU = (*event->getRanges());
    //int nTrackCandidates = 0;
    for (unsigned int idx = 0; idx <= *(modulesInGPU.nLowerModules); ++idx)
    {
        if (trackCandidatesInGPU.nTrackCandidates[idx] > 50000)
        {
            std::cout <<  " SDL::modulesInGPU->detIds[SDL::modulesInGPU->lowerModuleIndices[idx]]: " << modulesInGPU.detIds[idx] <<  std::endl;
            std::cout <<  " idx: " << idx <<  " trackCandidatesInGPU.nTrackCandidates[idx]: " << trackCandidatesInGPU.nTrackCandidates[idx] <<  std::endl;
            std::cout <<  " idx: " << idx <<  " tripletsInGPU.nTriplets[idx]: " << tripletsInGPU.nTriplets[idx] <<  std::endl;
            unsigned int i = idx;//modulesInGPU.lowerModuleIndices[idx];
            std::cout <<  " idx: " << idx <<  " i: " << i <<  " segmentsInGPU.nSegments[i]: " << segmentsInGPU.nSegments[i] <<  std::endl;
            int nMD = miniDoubletsInGPU.nMDs[2*idx]+miniDoubletsInGPU.nMDs[2*idx+1] ;
            std::cout <<  " idx: " << idx <<  " nMD: " << nMD <<  std::endl;
            int nHits = 0;
            nHits += rangesInGPU.hitRanges[4*idx+1] - rangesInGPU.hitRanges[4*idx] + 1;
            nHits += rangesInGPU.hitRanges[4*idx+3] - rangesInGPU.hitRanges[4*idx+2] + 1;
            std::cout <<  " idx: " << idx <<  " nHits: " << nHits <<  std::endl;
        }
    }
}
