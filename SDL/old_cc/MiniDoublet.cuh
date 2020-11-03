#ifndef MiniDoublet_h
#define MiniDoublet_h

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <array>
#include <tuple>
#include <math.h>
#include "Constants.h"
#include "Algo.h"
#include "PrintUtil.h"
#include "MathUtil.cuh"
#include "EndcapGeometry.h"
#include "TiltedGeometry.h"
#include "Hit.cuh"
#include "Module.cuh"
namespace SDL
{
    class Module;
    class Hit;
    class Segment;
}

namespace SDL
{
    class MiniDoublet
    {
        private:

            // TODO: Rename lower and upper as inner and outer

            // Lower is always the one closer to the beam position
            Hit* lowerHitPtr_;

            // Upper is always the one further away from the beam position
            Hit* upperHitPtr_;

            // Anchor hit is either Pixel hit (if available) or lower hit
            Hit* anchorHitPtr_;

            // Bits to flag whether this mini-doublet passes some algorithm
            int passAlgo_;

            // Some mini-doublet related reconstructon variables
            Hit lowerShiftedHit_;
            Hit upperShiftedHit_;
            float dz_;
            float shiftedDz_;
            float dphi_;
            float dphi_noshift_;
            float dphichange_;
            float dphichange_noshift_;

            float drdz_; //drdz of lower module
            float slopeForHitShifting_;

            // Pointers of segments containing this Mini-doublet as inner mini doublet
//            std::vector<Segment*> outwardSegmentPtrs;

            // Pointers of segments containing this Mini-doublet as outer mini doublet
//            std::vector<Segment*> inwardSegmentPtrs;

            float miniCut_;
            CUDA_HOSTDEV void setDerivedQuantities();

        public:
            CUDA_HOSTDEV MiniDoublet();
            CUDA_HOSTDEV MiniDoublet(const MiniDoublet&);
            CUDA_HOSTDEV MiniDoublet(Hit* lowerHit, Hit* upperHit);
            CUDA_HOSTDEV ~MiniDoublet();

            const std::vector<Segment*>& getListOfOutwardSegmentPtrs();
            const std::vector<Segment*>& getListOfInwardSegmentPtrs();

            void addOutwardSegmentPtr(Segment* sg);
            void addInwardSegmentPtr(Segment* sg);

            CUDA_HOSTDEV Hit* lowerHitPtr() const;
            CUDA_HOSTDEV Hit* upperHitPtr() const;
            CUDA_HOSTDEV Hit* anchorHitPtr() const;
            CUDA_HOSTDEV const int& getPassAlgo() const;
            CUDA_HOSTDEV const Hit& getLowerShiftedHit() const;
            CUDA_HOSTDEV const Hit& getUpperShiftedHit() const;
            CUDA_HOSTDEV const float& getDz() const;
            CUDA_HOSTDEV const float& getShiftedDz() const;
            CUDA_HOSTDEV const float& getDeltaPhi() const;
            CUDA_HOSTDEV const float& getDeltaPhiChange() const;
            CUDA_HOSTDEV const float& getDeltaPhiNoShift() const;
            CUDA_HOSTDEV const float& getDeltaPhiChangeNoShift() const;
            CUDA_HOSTDEV const float& getMiniCut() const;


            CUDA_HOSTDEV void setAnchorHit();
            CUDA_HOSTDEV void setLowerShiftedHit(float, float, float, int=-1);
            CUDA_HOSTDEV void setUpperShiftedHit(float, float, float, int=-1);
            CUDA_HOSTDEV void setDz(float);
            CUDA_HOSTDEV void setShiftedDz(float);
            CUDA_HOSTDEV void setDeltaPhi(float);
            CUDA_HOSTDEV void setDeltaPhiChange(float);
            CUDA_HOSTDEV void setDeltaPhiNoShift(float);
            CUDA_HOSTDEV void setDeltaPhiChangeNoShift(float);
            CUDA_HOSTDEV void setMiniCut(float);

            // return whether it passed the algorithm
            CUDA_HOSTDEV bool passesMiniDoubletAlgo(MDAlgo algo) const;

            // The function to run mini-doublet algorithm on a mini-doublet candidate
            CUDA_HOSTDEV void runMiniDoubletAlgo(MDAlgo algo, SDL::LogLevel logLevel=SDL::Log_Nothing);

            // The following algorithm does nothing and accepts the mini-doublet
            CUDA_HOSTDEV void runMiniDoubletAllCombAlgo();

            // The default algorithms;
            CUDA_HOSTDEV void runMiniDoubletDefaultAlgo(SDL::LogLevel logLevel);
            CUDA_HOSTDEV void runMiniDoubletDefaultAlgoBarrel(SDL::LogLevel logLevel);
            CUDA_HOSTDEV void runMiniDoubletDefaultAlgoEndcap(SDL::LogLevel logLevel);

            bool isIdxMatched(const MiniDoublet&) const;
            bool isAnchorHitIdxMatched(const MiniDoublet&) const;

            // cout printing
            friend std::ostream& operator<<(std::ostream& out, const MiniDoublet& md);
            friend std::ostream& operator<<(std::ostream& out, const MiniDoublet* md);

            // The math for the threshold cut value to apply between hits for mini-doublet
            // The main idea is to be consistent with 1 GeV minimum pt
            // Some residual effects such as tilt, multiple scattering, beam spots are considered
            //static float dPhiThreshold(const Hit&, const Module&);
            CUDA_HOSTDEV float dPhiThreshold(const Hit& lowerHit, const Module& module, const float dPhi = 0, const float dz = 1);

            // The math for shifting the pixel hit up or down along the PS module orientation (deprecated)
            static float fabsdPhiPixelShift(const Hit& lowerHit, const Hit& upperHit, const Module& lowerModule, SDL::LogLevel logLevel=SDL::Log_Nothing);

            // The math for shifting the strip hit up or down along the PS module orientation (deprecated)
            static float fabsdPhiStripShift(const Hit& lowerHit, const Hit& upperHit, const Module& lowerModule, SDL::LogLevel logLevel=SDL::Log_Nothing);

            // The math for shifting the strip hit up or dowCUDA_HOSTDEV n along the PS module orientation, returns new x, y and z position
            CUDA_HOSTDEV void shiftStripHits(const Hit& lowerHit, const Hit& upperHit, const Module& lowerModule, float* shiftedCoords, SDL::LogLevel logLevel=SDL::Log_Nothing);

            // The function to actually determine whether a pair of hits is a reco-ed mini doublet or not
            static bool isHitPairAMiniDoublet(const Hit& lowerHit, const Hit& upperHit, const Module& lowerModule, MDAlgo algo, SDL::LogLevel logLevel=SDL::Log_Nothing);

            // Condition that a module falls into "barrel"-logic of the mini-doublet algorithm
            static bool useBarrelLogic(const Module& lowerModule);

            // The function to determine transition region for inner most tilted layer
            static bool isNormalTiltedModules(const Module& lowerModule);

            // The function to determine transition region for inner most tilted layer (same as isNormalTiltedModules)
            CUDA_HOSTDEV static bool isTighterTiltedModules(const Module& lowerModule);

            // The function to determine gap
            CUDA_HOSTDEV static float moduleGapSize(const Module& lowerModule);

            //Function to set drdz so that we don't transport the tilted module map every time into the GPU, also
            //GPUs don't have STL yet, so we can't transport the map even if we wanted
            CUDA_HOSTDEV void setDrDz(float);

            CUDA_HOSTDEV void setLowerModuleSlope(float);


            
    };
}

#endif
