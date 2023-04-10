#include "simpleTCFit.h"
#include "circlefit/circlefit.h"

std::tuple<float, float, float, float, float, int> simpleTCFit(
    std::vector<unsigned int> hit_idx,
    std::vector<unsigned int> hit_type,
    int itc)
{

    // Output variables
    float pt, eta, phi, dxy, dz;
    int charge;

    // float arrays for hit positions
    const unsigned int nhits = hit_idx.size();
    reals X[nhits];
    reals Y[nhits];
    reals Z[nhits];
    reals R[nhits];

    // retrieve the hit positions
    for (unsigned int ihit = 0; ihit < nhits; ++ihit)
    {
        X[ihit] = hit_type[ihit] == 4 ? trk.ph2_x()[hit_idx[ihit]] : trk.pix_x()[hit_idx[ihit]];
        Y[ihit] = hit_type[ihit] == 4 ? trk.ph2_y()[hit_idx[ihit]] : trk.pix_y()[hit_idx[ihit]];
        Z[ihit] = hit_type[ihit] == 4 ? trk.ph2_z()[hit_idx[ihit]] : trk.pix_z()[hit_idx[ihit]];
        R[ihit] = sqrt(X[ihit] * X[ihit] + Y[ihit] * Y[ihit]);
    }

    //=========================================================================================
    //
    //
    // Transverse plance calculations (pt, phi, and dxy)
    //
    //
    //=========================================================================================

    // Circle fit using a package from some guy...
    // reference : Circle fit package from https://people.cas.uab.edu/~mosya/cl/CPPcircle.html

    // Input data to be passed to circle fitter
    Data data1(nhits, X, Y);

    // Circle fit results
    Circle circleIni; // Initial guess on the circle fit which is obtained from algebraic fit
    Circle circle; // Final fitted result via geometrical fit algorithms
    reals LambdaIni = 0.01; // hyperparameter for something called "Levenberg-Marquardt procedure"

    // Algebraic circle fits to first obtain a initial guess.
    circleIni = CircleFitByTaubin (data1);

    // This following fit is known to be stable for large radius (i.e. for high-pt)
    int code = CircleFitByChernovHoussam (data1,circleIni,LambdaIni,circle);
    // error code
    if ((code == 1)||(code==2)) cout << "\n Geometric circle by Chernov-Houssam did not converge. Iterations maxed out.\n";
    if (code == 3) cout << "\n Geometric circle by Chernov-Houssam did not converge. Fitting circle too big.\n";

    // -----------------
    // circle fit result
    // -----------------
    // center
    float center_x = circle.a;
    float center_y = circle.b;
    float angle_to_center = atan2(center_y, center_x);
    float dist_to_center = sqrt(circle.a * circle.a + circle.b * circle.b);
    // radius
    float radius = circle.r;
    float radius_x_projection_to_center = radius * cos(angle_to_center);
    float radius_y_projection_to_center = radius * sin(angle_to_center);
    // pt
    pt = radius * (2.99792458e-3 * 3.8);
    // point of closest approach
    float pca_x = center_x - radius_x_projection_to_center;
    float pca_y = center_y - radius_y_projection_to_center;
    dxy = (dist_to_center - radius);
    // charge
    float cross = -((X[1] - X[0]) * (Y[nhits - 1] - Y[0]) - (Y[1] - Y[0]) * (X[nhits - 1] - X[0]));
    charge = (cross > 0) - (cross < 0);

    // phi
    phi = SDL::CPU::MathUtil::Phi_mpi_pi(angle_to_center + (charge > 0 ? M_PI / 2. : -M_PI / 2.));

    //=========================================================================================
    //
    //
    // R-Z plance calculations (eta, and dz)
    //
    //
    //=========================================================================================

    // fitting straight line to obtain eta and dz
    auto g = new TGraph(nhits, Z, R);
    g->Fit("pol1", "Q");
    TF1* myfunc = g->GetFunction("pol1");
    // ax + b
    float b = myfunc->GetParameter(0);
    float a = myfunc->GetParameter(1);
    eta = ((a > 0) - ( a < 0)) * std::acosh(sqrt(a * a + 1) / abs(a));
    dz = (dxy - b) / a;

    return {pt, eta, phi, dxy, dz, charge};
}
