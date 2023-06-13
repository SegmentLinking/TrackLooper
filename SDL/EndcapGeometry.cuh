#ifndef EndcapGeometry_h
#define EndcapGeometry_h

#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "Constants.cuh"

namespace SDL
{
    class EndcapGeometry
    {
        private:
            std::map<unsigned int, float> avgr2s_;
            std::map<unsigned int, float> yls_; // lower hits
            std::map<unsigned int, float> sls_; // lower slope
            std::map<unsigned int, float> yus_; // upper hits
            std::map<unsigned int, float> sus_; // upper slope
            std::map<unsigned int, float> centroid_rs_; // centroid r
            std::map<unsigned int, float> centroid_phis_; // centroid phi
            std::map<unsigned int, float> centroid_zs_; // centroid z

        public:
            Buf<Acc, unsigned int> geoMapDetId_buf;
            Buf<Acc, float> geoMapPhi_buf;

            unsigned int nEndCapMap;

            EndcapGeometry(unsigned int sizef = endcap_size);
            EndcapGeometry(std::string filename, unsigned int sizef = endcap_size);
            ~EndcapGeometry();

            void load(std::string);

        void fillGeoMapArraysExplicit();
        void CreateGeoMapArraysExplicit();
        float getAverageR2(unsigned int detid);
        float getYInterceptLower(unsigned int detid);
        float getSlopeLower(unsigned int detid);
        float getYInterceptUpper(unsigned int detid);
        float getSlopeUpper(unsigned int detid);
        float getCentroidR(unsigned int detid);
        float getCentroidPhi(unsigned int detid);
        float getCentroidZ(unsigned int detid);

    };
    void freeEndCapMapMemory();
    extern EndcapGeometry endcapGeometry;
}

#endif
