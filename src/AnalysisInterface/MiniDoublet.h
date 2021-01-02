#ifndef ANALYSIS_INTERFACE_MINIDOUBLET_H
#define ANALYSIS_INTERFACE_MINIDOUBLET_H

#include <vector>
#include <map>
#include <tuple>
#include <memory>

#include "Module.h"
#include "Hit.h"

namespace SDL
{
    class Module;
    class Hit;

    class MiniDoublet
    {   
        private:
            float dz_;
            float drt_;
            float dphi_;
            float dphichange_;
            float dphinoshift_;
            float dphichangenoshift_;
            float dzCut_;
            float drtCut_;
            float miniCut_;

            std::shared_ptr<Hit> lowerHitPtr_;
            std::shared_ptr<Hit> upperHitPtr_;
            std::shared_ptr<Hit> anchorHitPtr_;
        public:
            MiniDoublet(float dz, float drt, float dphi, float dphichange, float dphinoshift, float dphichangenoshift, float dzCut, float drtCut, float miniCut, std::shared_ptr<Hit> lowerHitPtr, std::shared_ptr<Hit> upperHitPtr);
            void setAnchorHit();
            ~MiniDoublet();
            std::shared_ptr<Hit> lowerHitPtr() const;
            std::shared_ptr<Hit> upperHitPtr() const;
            std::shared_ptr<Hit> anchorHitPtr() const;
            const float& getDz() const;
            const float& getDrt() const;
            const float& getDeltaPhi() const;
            const float& getDeltaPhiChange() const;
            const float& getDeltaPhiNoShift() const;
            const float& getDeltaPhiChangeNoShift() const;
            const float& getDzCut() const;
            const float& getDrtCut() const;
            const float& getMiniCut() const;

    };
}
#endif
