#ifndef ANALYSIS_INTERFACE_H
#define ANALYSIS_INTERFACE_H

#include <vector>
#include <map>
#include <tuple>
#include <memory>

#include "Hit.h"
#include "MiniDoublet.h"
#include "Segment.h"
#include "SDL/Module.cuh"

namespace SDL
{
    class Hit;
    class MiniDoublet;
    class Segment;
    class Tracklet;
    class Triplet;
    class TrackCandidate;

    class Module
    {
        private:
            unsigned int detId_;
            unsigned int partnerDetId_;
            short layer_;
            short ring_;
            short rod_;
            short module_;
            bool isInverted_;
            bool isLower_;
        public:
            enum SubDet
            {
                InnerPixel = 0,
                Barrel = 5,
                Endcap = 4
            };
        private:
            SubDet subdet_;
        public:
            enum ModuleType
            {
                PS,
                TwoS,
                PixelModule
            };
        private:

            ModuleType moduleType_;
        public:
            enum ModuleLayerType
            {
                Pixel,
                Strip,
                InnerPixelLayer
            };
        private:
            ModuleLayerType moduleLayerType_;
        public:
            enum Side
            {
                NegZ = 1,
                PosZ = 2,
                Center = 3
            };
        private:
            Side side_;

            std::vector<std::shared_ptr<Hit>> hits_;
            std::vector<std::shared_ptr<MiniDoublet>> miniDoublets_;
            std::vector<std::shared_ptr<Segment>> segments_;
            std::vector<std::shared_ptr<Tracklet>> tracklets_;
            std::vector<std::shared_ptr<Triplet>> triplets_;
            std::vector<std::shared_ptr<TrackCandidate>> trackCandidates_;
            //to accommodate the fact that we will overshoot existing array limits and need another way to count object multiplicities
            unsigned int nMiniDoublets_;
            unsigned int nSegments_;
            unsigned int nTracklets_;
            unsigned int nTriplets_;
            unsigned int nTrackCandidates_;
            unsigned int nTrackCandidatesT4T4_;
            unsigned int nTrackCandidatesT4T3_;
            unsigned int nTrackCandidatesT3T4_;

        public:
            Module();
            Module(unsigned int detId);
            Module(unsigned int detId, short layer, short ring, short rod, short module, bool isInverted, bool isLower, short subdet, SDL::ModuleType moduleType, SDL::ModuleLayerType moduleLayerType, short side);
            ~Module();

            // accessor functions
            const unsigned int& detId() const;
            const unsigned int& partnerDetId() const;
            const SDL::Module::SubDet& subdet() const;
            const short& side() const;
            const short& layer() const;
            const short& rod() const;
            const short& ring() const;
            const short& module() const;
            const bool& isLower() const;
            const bool& isInverted() const;
            const ModuleType& moduleType() const;
            const ModuleLayerType& moduleLayerType() const;
            const std::vector<std::shared_ptr<Hit>>& getHitPtrs() const;
            const std::vector<std::shared_ptr<MiniDoublet>>& getMiniDoubletPtrs() const;
            const std::vector<std::shared_ptr<Segment>>& getSegmentPtrs() const;
            const std::vector<std::shared_ptr<Tracklet>>& getTrackletPtrs() const;
            const std::vector<std::shared_ptr<Triplet>>& getTripletPtrs() const;
            const std::vector<std::shared_ptr<TrackCandidate>>& getTrackCandidatePtrs() const;

            const int getNumberOfMiniDoublets() const;
            const int getNumberOfSegments() const;
            const int getNumberOfTracklets() const;
            const int getNumberOfTriplets() const;
            const int getNumberOfTrackCandidates() const;
            const int getNumberOfTrackCandidatesT4T4() const;
            const int getNumberOfTrackCandidatesT4T3() const;
            const int getNumberOfTrackCandidatesT3T4() const;

            void setNumberOfMiniDoublets(unsigned int);
            void setNumberOfSegments(unsigned int);
            void setNumberOfTracklets(unsigned int);
            void setNumberOfTriplets(unsigned int);
            void setNumberOfTrackCandidates(unsigned int);
            void setNumberOfTrackCandidatesT4T4(unsigned int);
            void setNumberOfTrackCandidatesT4T3(unsigned int);
            void setNumberOfTrackCandidatesT3T4(unsigned int);


            // static functions to parse detId
            static unsigned short parseSubdet(unsigned int);
            static unsigned short parseSide(unsigned int);
            static unsigned short parseLayer(unsigned int);
            static unsigned short parseRod(unsigned int);
            static unsigned short parseRing(unsigned int);
            static unsigned short parseModule(unsigned int);
            static unsigned short parseIsLower(unsigned int);
            static bool parseIsInverted(unsigned int);
            static unsigned int parsePartnerDetId(unsigned int);
            static ModuleType parseModuleType(unsigned int);
            static ModuleLayerType parseModuleLayerType(unsigned int);
            void setDerivedQuantities();


            void addHit(std::shared_ptr<Hit> hit);
            void addMiniDoublet(std::shared_ptr<MiniDoublet> md);
            void addSegment(std::shared_ptr<Segment> sg);
            void addTracklet(std::shared_ptr<Tracklet> tp);
            void addTriplet(std::shared_ptr<Triplet> tp);
            void addTrackCandidate(std::shared_ptr<TrackCandidate> tc);
    };

}
#endif
