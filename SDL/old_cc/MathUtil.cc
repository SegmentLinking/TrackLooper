#include "MathUtil.h"

float SDL::MathUtil::Phi_mpi_pi(float x)
{
    if (std::isnan(x))
    {
        std::cout << "SDL::MathUtil::Phi_mpi_pi() function called with NaN" << std::endl;
        return x;
    }

    while (x >= M_PI)
        x -= 2. * M_PI;

    while (x < -M_PI)
        x += 2. * M_PI;

    return x;

}

float SDL::MathUtil::ATan2(float y, float x)
{
    if (x != 0) return  atan2(y, x);
    if (y == 0) return  0;
    if (y >  0) return  M_PI / 2;
    else        return -M_PI / 2;
}

float SDL::MathUtil::ptEstimateFromDeltaPhiChangeAndRt(float dPhiChange, float rt)
{
    return rt * 2.99792458e-3 * 3.8 / 2. / std::sin(dPhiChange);
}

float SDL::MathUtil::ptEstimateFromRadius(float radius)
{
    return 2.99792458e-3 * 3.8 * radius;
}

float SDL::MathUtil::dphiEstimateFromPtAndRt(float pt, float rt)
{
    return std::asin(rt / (2 * pt / (2.99792458e-3 * 3.8)));
}

SDL::Hit SDL::MathUtil::getCenterFromThreePoints(SDL::Hit& hitA, SDL::Hit& hitB, SDL::Hit& hitC)
{

    //       C
    //
    //
    //
    //    B           d <-- find this point that makes the arc that goes throw a b c
    //
    //
    //     A

    // Steps:
    // 1. Calculate mid-points of lines AB and BC
    // 2. Find slopes of line AB and BC
    // 3. construct a perpendicular line between AB and BC
    // 4. set the two equations equal to each other and solve to find intersection

    float xA = hitA.x();
    float yA = hitA.y();
    float xB = hitB.x();
    float yB = hitB.y();
    float xC = hitC.x();
    float yC = hitC.y();

    float x_mid_AB = (xA + xB) / 2.;
    float y_mid_AB = (yA + yB) / 2.;

    float x_mid_BC = (xB + xC) / 2.;
    float y_mid_BC = (yB + yC) / 2.;

    bool slope_AB_inf = (xB - xA) == 0;
    bool slope_BC_inf = (xC - xB) == 0;

    bool slope_AB_zero = (yB - yA) == 0;
    bool slope_BC_zero = (yC - yB) == 0;

    float slope_AB = slope_AB_inf ? 0 : (yB - yA) / (xB - xA);
    float slope_BC = slope_BC_inf ? 0 : (yC - yB) / (xC - xB);

    float slope_perp_AB = slope_AB_inf or slope_AB_zero ? 0. : -1. / (slope_AB);
    float slope_perp_BC = slope_BC_inf or slope_BC_zero ? 0. : -1. / (slope_BC);

    if ((slope_AB - slope_BC) == 0)
    {
        std::cout <<  " slope_AB_zero: " << slope_AB_zero <<  std::endl;
        std::cout <<  " slope_BC_zero: " << slope_BC_zero <<  std::endl;
        std::cout <<  " slope_AB_inf: " << slope_AB_inf <<  std::endl;
        std::cout <<  " slope_BC_inf: " << slope_BC_inf <<  std::endl;
        std::cout <<  " slope_AB: " << slope_AB <<  std::endl;
        std::cout <<  " slope_BC: " << slope_BC <<  std::endl;
        std::cout << hitA << std::endl;
        std::cout << hitB << std::endl;
        std::cout << hitC << std::endl;
        std::cout << "SDL::MathUtil::getCenterFromThreePoints() function the three points are in straight line!" << std::endl;
        return SDL::Hit();
    }

    float x = (slope_AB * slope_BC * (yA - yC) + slope_BC * (xA + xB) - slope_AB * (xB + xC)) / (2. * (slope_BC - slope_AB));
    float y = slope_perp_AB * (x - x_mid_AB) + y_mid_AB;

    return SDL::Hit(x, y, 0);

}

float SDL::MathUtil::angleCorr(float dr, float pt, float angle)
{
    const float kRinv1GeVf = (2.99792458e-3 * 3.8);
    const float k2Rinv1GeVf = kRinv1GeVf / 2.;
    const float sinAlphaMax = 0.95;
    return copysign(std::asin(std::min(dr * k2Rinv1GeVf / std::abs(pt), sinAlphaMax)), angle);
}
