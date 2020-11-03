#include "GeometryUtil.cuh"

SDL::Hit SDL::GeometryUtil::stripEdgeHit(const SDL::Hit& recohit, bool isup)
{
    const SDL::Module& module = recohit.getModule();
    
    if(module.moduleLayerType() != SDL::Module::Strip)
    SDL::cout << "Warning: stripEdgeHit() is asked on a hit that is not strip hit" << std::endl;

    const unsigned int& detid = module.detId();

    float phi = SDL::endcapGeometry.getCentroidPhi(detid); // Only need one slope

    float sign = isup ? 1. : -1;

    SDL::Hit edge_hitvec(sign * 2.5 * cos(phi), sign * 2.5 * sin(phi), 0);

    // edge_hitvec.setModule(&module);

    edge_hitvec += recohit;

    return edge_hitvec;
}

SDL::Hit SDL::GeometryUtil::stripHighEdgeHit(const SDL::Hit& recohit)
{
    return stripEdgeHit(recohit, true);
}

SDL::Hit SDL::GeometryUtil::stripLowEdgeHit(const SDL::Hit& recohit)
{
    return stripEdgeHit(recohit, false);
}

