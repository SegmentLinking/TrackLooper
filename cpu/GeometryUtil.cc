#include "GeometryUtil.h"

SDL::CPU::Hit SDL::CPU::GeometryUtil::stripEdgeHit(const SDL::CPU::Hit& recohit, bool isup)
{
    const SDL::CPU::Module& module = recohit.getModule();

    if (module.moduleLayerType() != SDL::CPU::Module::Strip)
        SDL::CPU::cout << "Warning: stripEdgeHit() is asked on a hit that is not strip hit" << std::endl;

    const unsigned int& detid = module.detId();

    float phi = SDL::endcapGeometry.getCentroidPhi(detid); // Only need one slope

    float sign = isup ? 1. : -1;

    SDL::CPU::Hit edge_hitvec(sign * 2.5 * cos(phi), sign * 2.5 * sin(phi), 0);

    // edge_hitvec.setModule(&module);

    edge_hitvec += recohit;

    return edge_hitvec;
}

SDL::CPU::Hit SDL::CPU::GeometryUtil::stripHighEdgeHit(const SDL::CPU::Hit& recohit)
{
    return stripEdgeHit(recohit, true);
}

SDL::CPU::Hit SDL::CPU::GeometryUtil::stripLowEdgeHit(const SDL::CPU::Hit& recohit)
{
    return stripEdgeHit(recohit, false);
}

