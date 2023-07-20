#include "EndcapGeometry.cuh"

SDL::EndcapGeometry SDL::endcapGeometry;

SDL::EndcapGeometry::EndcapGeometry(unsigned int sizef) :
    geoMapDetId_buf(allocBufWrapper<unsigned int>(devAcc, sizef)),
    geoMapPhi_buf(allocBufWrapper<float>(devAcc, sizef))
{
}

SDL::EndcapGeometry::EndcapGeometry(std::string filename, unsigned int sizef) :
    geoMapDetId_buf(allocBufWrapper<unsigned int>(devAcc, sizef)),
    geoMapPhi_buf(allocBufWrapper<float>(devAcc, sizef))
{
    load(filename);
}

SDL::EndcapGeometry::~EndcapGeometry()
{
}

void SDL::EndcapGeometry::load(std::string filename)
{
    avgr2s_.clear();
    yls_.clear();
    sls_.clear();
    yus_.clear();
    sus_.clear();
    centroid_rs_.clear();
    centroid_phis_.clear();
    centroid_zs_.clear();

    std::ifstream ifile;
    ifile.open(filename.c_str());
    std::string line;

    while (std::getline(ifile, line))
    {

        unsigned int detid;
        float avgr2;
        float yl;
        float sl;
        float yh;
        float sh;
        float cr;
        float cp;
        float cz;

        std::stringstream ss(line);

        ss >> detid >> avgr2 >> yl >> sl >> yh >> sh >> cr >> cp >> cz;

        // std::cout <<  " detid: " << detid <<  " avgr2: " << avgr2 <<  " yl: " << yl <<  " sl: " << sl <<  " yh: " << yh <<  " sh: " << sh <<  std::endl;

        avgr2s_[detid] = avgr2;
        yls_[detid] = yl;
        sls_[detid] = sl;
        yus_[detid] = yh;
        sus_[detid] = sh;
        centroid_rs_[detid] = cp;
        centroid_phis_[detid] = cr;
        centroid_zs_[detid] = cz;
    }

    fillGeoMapArraysExplicit();
}

void SDL::EndcapGeometry::fillGeoMapArraysExplicit()
{
    QueueAcc queue(devAcc);

    int phi_size = centroid_phis_.size();

    // Temporary check for endcap initialization.
    if(phi_size != endcap_size) {
        std::cerr << "\nError: phi_size and endcap_size are not equal.\n";
        std::cerr << "phi_size: " << phi_size << ", endcap_size: " << endcap_size << "\n";
        std::cerr << "Please change endcap_size in Constants.cuh to make it equal to phi_size.\n";
        throw std::runtime_error("Mismatched sizes");
    }

    // Allocate buffers on host
    auto mapPhi_host_buf = allocBufWrapper<float>(devHost, phi_size);
    auto mapDetId_host_buf = allocBufWrapper<unsigned int>(devHost, phi_size);

    // Access the raw pointers of the buffers
    float* mapPhi = alpaka::getPtrNative(mapPhi_host_buf);
    unsigned int* mapDetId = alpaka::getPtrNative(mapDetId_host_buf);

    unsigned int counter = 0;
    for(auto it = centroid_phis_.begin(); it != centroid_phis_.end(); ++it)
    {
        unsigned int detId = it->first;
        float Phi = it->second;
        mapPhi[counter] = Phi;
        mapDetId[counter] = detId;
        counter++;
    }

    nEndCapMap = counter;

    // Copy data from host to device buffers
    alpaka::memcpy(queue, geoMapPhi_buf, mapPhi_host_buf, phi_size);
    alpaka::memcpy(queue, geoMapDetId_buf, mapDetId_host_buf, phi_size);
    alpaka::wait(queue);
}

float SDL::EndcapGeometry::getAverageR2(unsigned int detid)
{
    return avgr2s_[detid];
}

float SDL::EndcapGeometry::getYInterceptLower(unsigned int detid)
{
    return yls_[detid];
}

float SDL::EndcapGeometry::getSlopeLower(unsigned int detid)
{
    return sls_[detid];
}

float SDL::EndcapGeometry::getYInterceptUpper(unsigned int detid)
{
    return yus_[detid];
}

float SDL::EndcapGeometry::getSlopeUpper(unsigned int detid)
{
    return sus_[detid];
}

float SDL::EndcapGeometry::getCentroidR(unsigned int detid)
{
    return centroid_rs_[detid];
}

float SDL::EndcapGeometry::getCentroidPhi(unsigned int detid)
{
    return centroid_phis_[detid];
}

float SDL::EndcapGeometry::getCentroidZ(unsigned int detid)
{
    return centroid_zs_[detid];
}
