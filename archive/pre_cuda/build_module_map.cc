#include "build_module_map.h"

void build_module_map()
{

    // connection information
    std::ofstream module_connection_log_output;
    module_connection_log_output.open("data/conn.txt");

    module_connection_log_output << "module_connection_raw_data = [" << std::endl;;

    // Looping input file
    while (ana.looper.nextEvent())
    {

        if (not (goodEvent()))
            continue;

        printModuleConnections(module_connection_log_output);

    }

    module_connection_log_output << "]";
}

void printModuleConnections(std::ofstream& ostrm)
{
    // *****************************************************
    // Print module -> module connection info from sim track
    // *****************************************************

    // Loop over sim-tracks and per sim-track aggregate good hits (i.e. matched with particle ID)
    // and only use those hits, and run mini-doublet reco algorithm on the sim-track-matched-reco-hits
    for (unsigned int isimtrk = 0; isimtrk < trk.sim_q().size(); ++isimtrk)
    {

        // Select only muon tracks
        if (abs(trk.sim_pdgId()[isimtrk]) != 13)
            continue;

        if (trk.sim_pt()[isimtrk] < 1)
            continue;

        std::vector<int> detids;
        std::vector<float> simhit_xs;
        std::vector<float> simhit_ys;
        std::vector<float> simhit_zs;
        std::vector<float> ps;

        // loop over the simulated hits
        for (auto& simhitidx : trk.sim_simHitIdx()[isimtrk])
        {

            // Select only the hits in the outer tracker
            if (not (trk.simhit_subdet()[simhitidx] == 4 or trk.simhit_subdet()[simhitidx] == 5))
                continue;

            int simhit_particle = trk.simhit_particle()[simhitidx];
            int detid = trk.simhit_detId()[simhitidx];
            int layer = trk.simhit_layer()[simhitidx];
            float x = trk.simhit_x()[simhitidx];
            float y = trk.simhit_y()[simhitidx];
            float z = trk.simhit_z()[simhitidx];
            float r3 = sqrt(x*x + y*y + z*z);
            float px = trk.simhit_px()[simhitidx];
            float py = trk.simhit_py()[simhitidx];
            float pz = trk.simhit_pz()[simhitidx];
            float p = sqrt(px*px + py*py + pz*pz);
            float rdotp = x*px + y*py + z*pz;
            int subdet = trk.simhit_subdet()[simhitidx];
            int trkidx = trk.simhit_simTrkIdx()[simhitidx];
            SDL::Module module = SDL::Module(detid);

            // Select only the sim hits that is matched to the muon
            if (abs(simhit_particle) != 13)
                continue;

            // Stop if the particle momentum is incoming
            if ((rdotp) < 0)
                break;

            // Stop if the particle momentum reaches below 0.9 GeV
            if (sqrt(px*px + py*py) < 0.9)
                break;

            // Stop if the momentum loss is bigger than 2%
            if (ps.size() > 0)
            {
                float loss = fabs(ps.back() - p) / ps.back();
                if (loss > 0.02)
                    break;
            }

            // Aggregate isLower modules only
            if (module.isLower())
            {
                detids.push_back(detid);
                simhit_xs.push_back(x);
                simhit_ys.push_back(y);
                simhit_zs.push_back(z);
                ps.push_back(p);
            }

        }

        if (detids.size() > 0)
        {

            ostrm << "[";
            for (auto&& [detid, x, y, z] : iter::zip(detids, simhit_xs, simhit_ys, simhit_zs))
            {
                ostrm << TString::Format("[%d, %f, %f, %f], ",detid, x, y, z);
            }
            ostrm << "]," << std::endl;

        }

    }
}
