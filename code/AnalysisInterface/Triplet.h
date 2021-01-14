#ifndef ANALYSIS_INTERFACE_TRIPLET_H
#define ANALYSIS_INTERFACE_TRIPLET_H

#include <vector>
#include <map>
#include <tuple>

#include "Module.h"
#include "TrackletBase.h"

namespace SDL
{
    class Module;
    class Triplet;
    class Segment;
    class TrackletBase;
}

namespace SDL
{

    class Triplet : public TrackletBase
    {
        private:
            float zOut_;
            float rtOut_;
            float dPhiPos_;
            float dPhi_;
            float betaIn_;
            float betaOut_;

            float zLo_;
            float zHi_;
            float zLoPointed_;
            float zHiPointed_;
            float sdlCut_;
            float betaInCut_;
            float betaOutCut_;
            float deltaBetaCut_;
            float rtLo_;
            float rtHi_;
            float kZ_;


        public:
            Triplet(float zOut, float rtOut, float dPhiPos, float dPhi, float betaIn, float betaOut, float zLo, float zHi, float zLoPointed, float zHiPointed, float sdlCut, float betaInCut, float betaOutCut, float deltaBetaCut, float rtLo, float rtHi, float kZ, std::shared_ptr<Segment> innerSegment, std::shared_ptr<Segment> outerSegment);
            ~Triplet();
    
 	    const float& getZOut() const;
	    const float& getRtOut() const;
	    const float& getDeltaPhiPos() const;
	    const float& getDeltaPhi() const;
        const float getDeltaBeta() const;
        const float& getBetaIn() const;
        const float& getBetaOut() const;

	    const float& getZLo() const;
	    const float& getRtLo() const;
	    const float& getZHi() const;
	    const float& getRtHi() const;
	    const float& getSDLCut() const;
 	    const float& getBetaInCut() const;
	    const float& getBetaOutCut() const;
	    const float& getDeltaBetaCut() const;
	    const float& getKZ() const;
 	    const float& getZLoPointed() const;
	    const float& getZHiPointed() const;


    };
}
#endif
