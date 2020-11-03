#ifndef SDLMath_h
#define SDLMath_h
#include <tuple>
#include <vector>

namespace SDLMath
{
    class Helix
    {
        public:
            std::vector<float> center_;
            float radius_;
            float phi_;
            float lam_;
            float charge_;

            Helix(std::vector<float> c, float r, float p, float l, float q)
            {
                center_ = c;
                radius_ = r;
                phi_ = Phi_mpi_pi(p);
                lam_ = l;
                charge_ = q;
            }

            Helix(float pt, float eta, float phi, float vx, float vy, float vz, float charge)
            {
                // Radius based on pt
                radius_ = pt / (2.99792458e-3 * 3.8);
                phi_ = phi;
                charge_ = charge;

                // reference point vector which for sim track is the vertex point
                float ref_vec_x = vx;
                float ref_vec_y = vy;
                float ref_vec_z = vz;

                // The reference to center vector
                float inward_radial_vec_x = charge_ * radius_ *  sin(phi_);
                float inward_radial_vec_y = charge_ * radius_ * -cos(phi_);
                float inward_radial_vec_z = 0;

                // Center point
                float center_vec_x = ref_vec_x + inward_radial_vec_x;
                float center_vec_y = ref_vec_y + inward_radial_vec_y;
                float center_vec_z = ref_vec_z + inward_radial_vec_z;
                center_.push_back(center_vec_x);
                center_.push_back(center_vec_y);
                center_.push_back(center_vec_z);

                // Lambda
                lam_ = copysign(M_PI / 2. - 2. * atan(exp(-abs(eta))), eta);
            }

            const std::vector<float> center() { return center_; }
            const float radius() { return radius_; }
            const float phi() { return phi_; }
            const float lam() { return lam_; }
            const float charge() { return charge_; }

            float Phi_mpi_pi(float phi)
            {
                float f = phi;
                while (f >= M_PI) f -= 2. * M_PI;
                while (f < -M_PI) f += 2. * M_PI;
                return f;
            }

            std::tuple<float, float, float, float> get_helix_point(float t)
            {
                float x = center()[0] - charge() * radius() * sin(phi() - (charge()) * t);
                float y = center()[1] + charge() * radius() * cos(phi() - (charge()) * t);
                float z = center()[2] +                  radius() * tan(lam()) * t;
                float r = sqrt(x*x + y*y);
                return std::make_tuple(x, y, z, r);
            }

            float infer_t(const std::vector<float> point)
            {
                // Solve for t based on z position
                float t = (point[2] - center()[2]) / (radius() * tan(lam()));
                return t;
            }

            float compare_radius(const std::vector<float> point)
            {
                float t = infer_t(point);
                auto [x, y, z, r] = get_helix_point(t);
                float point_r = sqrt(point[0]*point[0] + point[1]*point[1]);
                return (point_r - r);
            }

            float compare_xy(const std::vector<float> point)
            {
                float t = infer_t(point);
                auto [x, y, z, r] = get_helix_point(t);
                float xy_dist = sqrt(pow(point[0] - x, 2) + pow(point[1] - y, 2));
                return xy_dist;
            }

    };

}
#endif
