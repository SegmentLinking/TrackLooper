#ifndef ANALYSIS_INTERFACE_HIT_H
#define ANALYSIS_INTERFACE_HIT_H

#include <vector>
#include <map>
#include <tuple>
#include <memory>

#include "Module.h"

namespace SDL
{
    class Module;
    class Hit
    {  
        private:
            int idx_;
            float x_;
            float y_;
            float z_;
            float phi_;
            float rt_;
            float r3_;
            float eta_;
            std::shared_ptr<Hit> hit_high_edge_;
            std::shared_ptr<Hit> hit_low_edge_;
            std::shared_ptr<Module> modulePtr_;

        public:
            Hit(float x, float y, float z, float phi, float rt, int idx, std::shared_ptr<Module> modulePtr);
            Hit(float x, float y, float z);
            ~Hit();
            // accessor functions
            const int& idx() const;
            const float& x() const;
            const float& y() const;
            const float& z() const;
            const float& rt() const;
            const float& phi() const;
            const float& eta() const;
            Module& getModule() const;

            // Set the boundary hits where the hits are shifted
            const std::shared_ptr<Hit> getHitHighEdgePtr() const;
            const std::shared_ptr<Hit> getHitLowEdgePtr() const;
            void setHighEdgePtr(std::shared_ptr<Hit> hitHighEdge);
            void setLowEdgePtr(std::shared_ptr<Hit> hitLowEdge);
    };

}
#endif
