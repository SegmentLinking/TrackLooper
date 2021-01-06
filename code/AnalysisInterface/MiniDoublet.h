#ifndef ANALYSIS_INTERFACE_MINIDOUBLET_H
#define ANALYSIS_INTERFACE_MINIDOUBLET_H

#include <vector>
#include <map>
#include <tuple>

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
            float dphi_;
            float dphichange_;
            float dphinoshift_;
            float dphichangenoshift_;
            Hit* lowerHitPtr_;
            Hit* upperHitPtr_;
            Hit* anchorHitPtr_;
        public:
            MiniDoublet(float dz, float dphi, float dphichange, float dphinoshift, float dphichangenoshift, Hit* lowerHitPtr, Hit* upperHitPtr);
            void setAnchorHit();
            ~MiniDoublet();
            Hit* lowerHitPtr() const;
            Hit* upperHitPtr() const;
            Hit* anchorHitPtr() const;
            const float& getDz() const;
            const float& getDeltaPhi() const;
            const float& getDeltaPhiChange() const;
            const float& getDeltaPhiNoShift() const;
            const float& getDeltaPhiChangeNoShift() const;

    };
}
#endif
