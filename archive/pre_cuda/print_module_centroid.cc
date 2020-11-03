#include "print_module_centroid.h"

void print_module_centroid()
{

    // connection information
    std::ofstream module_connection_log_output;
    module_connection_log_output.open("data/centroid.txt");

    std::vector<unsigned int> detids;
    std::map<unsigned int, int> detid_count;
    std::map<unsigned int, float> detid_x;
    std::map<unsigned int, float> detid_y;
    std::map<unsigned int, float> detid_z;

    // All of the individual hit data
    std::map<int, std::vector<std::vector<float>>> module_simhits;

    ana.tx->createBranch<int>("detId");
    ana.tx->createBranch<int>("subdet");
    ana.tx->createBranch<int>("side");
    ana.tx->createBranch<int>("layer");
    ana.tx->createBranch<int>("rod");
    ana.tx->createBranch<int>("module");
    ana.tx->createBranch<int>("ring");
    ana.tx->createBranch<int>("isPS");
    ana.tx->createBranch<int>("isStrip");
    ana.tx->createBranch<vector<float>>("x");
    ana.tx->createBranch<vector<float>>("y");
    ana.tx->createBranch<vector<float>>("z");

    // Looping input file
    while (ana.looper.nextEvent())
    {

        if (not (goodEvent()))
            continue;

        for (auto&& [isimhit, detid] : iter::enumerate(trk.simhit_detId()))
        {
            if (not (trk.simhit_subdet()[isimhit] == 4 or trk.simhit_subdet()[isimhit] == 5))
                continue;
            detid_count[detid]++;
            detid_x[detid] += trk.simhit_x()[isimhit];
            detid_y[detid] += trk.simhit_y()[isimhit];
            detid_z[detid] += trk.simhit_z()[isimhit];
            module_simhits[detid].push_back({trk.simhit_x()[isimhit], trk.simhit_y()[isimhit], trk.simhit_z()[isimhit]});
            if (std::find(detids.begin(), detids.end(), detid) == detids.end())
                detids.push_back(detid);
        }
    }

    for (auto&& detid : iter::sorted(detids))
    {
        SDL::Module module(detid);
        module_connection_log_output << detid << ",";
        module_connection_log_output << detid_x[detid]/detid_count[detid] << ",";
        module_connection_log_output << detid_y[detid]/detid_count[detid] << ",";
        module_connection_log_output << detid_z[detid]/detid_count[detid] << ",";
        module_connection_log_output << detid_count[detid] << ",";
        module_connection_log_output << module.layer() + (module.subdet() == 4)*6;
        module_connection_log_output << std::endl;

        ana.tx->setBranch<int>("detId", detid);
        ana.tx->setBranch<int>("subdet", module.subdet());
        ana.tx->setBranch<int>("side", module.side());
        ana.tx->setBranch<int>("layer", module.layer());
        ana.tx->setBranch<int>("rod", module.rod());
        ana.tx->setBranch<int>("module", module.module());
        ana.tx->setBranch<int>("ring", module.ring());
        ana.tx->setBranch<int>("isPS", module.moduleType() == SDL::Module::PS);
        ana.tx->setBranch<int>("isStrip", module.moduleLayerType() == SDL::Module::Strip);

        for (auto& hit_coord : module_simhits[detid])
        {
            ana.tx->pushbackToBranch<float>("x", hit_coord[0]);
            ana.tx->pushbackToBranch<float>("y", hit_coord[1]);
            ana.tx->pushbackToBranch<float>("z", hit_coord[2]);
        }

        ana.tx->fill();
        ana.tx->clear();
    }

    // Write to tfile
    ana.output_tfile->cd();

    // Writing ttree output to file
    ana.output_ttree->Write();

    // The below can be sometimes crucial
    delete ana.output_tfile;

}

