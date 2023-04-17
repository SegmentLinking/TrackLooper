#ifndef TrackCandidate_cuh
#define TrackCandidate_cuh

#include "Constants.cuh"
#include "Triplet.cuh"
#include "Segment.cuh"
#include "MiniDoublet.cuh"
#include "PixelTriplet.cuh"
#include "Quintuplet.cuh"
#include "Module.cuh"
#include "Hit.cuh"

namespace SDL
{
    struct trackCandidates
    {
        short* trackCandidateType; // 4-T5 5-pT3 7-pT5 8-pLS
        unsigned int* directObjectIndices; // Will hold direct indices to each type containers
        unsigned int* objectIndices; // Will hold tracklet and  triplet indices - check the type!!
        int* nTrackCandidates;
        int* nTrackCandidatespT3;
        int* nTrackCandidatespT5;
        int* nTrackCandidatespLS;
        int* nTrackCandidatesT5;

        uint8_t* logicalLayers;
        unsigned int* hitIndices;
        int* pixelSeedIndex;
        uint16_t* lowerModuleIndices;

        FPX* centerX;
        FPX* centerY;
        FPX* radius;

        trackCandidates();
        ~trackCandidates();
        void freeMemory(cudaStream_t stream);
        void freeMemoryCache();
        void resetMemory(unsigned int maxTrackCandidates,cudaStream_t stream);
    };

    void createTrackCandidatesInExplicitMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int maxTrackCandidates,cudaStream_t stream);

    ALPAKA_FN_ACC void addpLSTrackCandidateToMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int trackletIndex, unsigned int trackCandidateIndex, uint4 hitIndices, int pixelSeedIndex)
    {
        trackCandidatesInGPU.trackCandidateType[trackCandidateIndex] = 8;
        trackCandidatesInGPU.directObjectIndices[trackCandidateIndex] = trackletIndex;
        trackCandidatesInGPU.pixelSeedIndex[trackCandidateIndex] = pixelSeedIndex;

        trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex] = trackletIndex;
        trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex + 1] = trackletIndex;

        trackCandidatesInGPU.hitIndices[14 * trackCandidateIndex + 0] = hitIndices.x; // Order explanation in https://github.com/SegmentLinking/TrackLooper/issues/267
        trackCandidatesInGPU.hitIndices[14 * trackCandidateIndex + 1] = hitIndices.z;
        trackCandidatesInGPU.hitIndices[14 * trackCandidateIndex + 2] = hitIndices.y;
        trackCandidatesInGPU.hitIndices[14 * trackCandidateIndex + 3] = hitIndices.w;
    };

    ALPAKA_FN_ACC void addTrackCandidateToMemory(struct trackCandidates& trackCandidatesInGPU, short trackCandidateType, unsigned int innerTrackletIndex, unsigned int outerTrackletIndex, uint8_t* logicalLayerIndices, uint16_t* lowerModuleIndices, unsigned int* hitIndices, int pixelSeedIndex, float centerX, float centerY, float radius, unsigned int trackCandidateIndex, unsigned int directObjectIndex)
    {
        trackCandidatesInGPU.trackCandidateType[trackCandidateIndex] = trackCandidateType;
        trackCandidatesInGPU.directObjectIndices[trackCandidateIndex] = directObjectIndex;
        trackCandidatesInGPU.pixelSeedIndex[trackCandidateIndex] = pixelSeedIndex;

        trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex] = innerTrackletIndex;
        trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex + 1] = outerTrackletIndex;
        
        size_t limits = trackCandidateType == 7 ? 7 : 5;

        //send the starting pointer to the logicalLayer and hitIndices
        for(size_t i = 0; i < limits; i++)
        {
            trackCandidatesInGPU.logicalLayers[7 * trackCandidateIndex + i] = logicalLayerIndices[i];
            trackCandidatesInGPU.lowerModuleIndices[7 * trackCandidateIndex + i] = lowerModuleIndices[i];
        }
        for(size_t i = 0; i < 2 * limits; i++)
        {
            trackCandidatesInGPU.hitIndices[14 * trackCandidateIndex + i] = hitIndices[i];
        }
        trackCandidatesInGPU.centerX[trackCandidateIndex] = __F2H(centerX);
        trackCandidatesInGPU.centerY[trackCandidateIndex] = __F2H(centerY);
        trackCandidatesInGPU.radius[trackCandidateIndex]  = __F2H(radius);
    };

    ALPAKA_FN_ACC ALPAKA_FN_INLINE int checkPixelHits(unsigned int ix, unsigned int jx, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::hits& hitsInGPU)
    {
        int phits1[4] = {-1,-1,-1,-1};
        int phits2[4] = {-1,-1,-1,-1};
        phits1[0] = hitsInGPU.idxs[mdsInGPU.anchorHitIndices[segmentsInGPU.mdIndices[2*ix]]];
        phits1[1] = hitsInGPU.idxs[mdsInGPU.anchorHitIndices[segmentsInGPU.mdIndices[2*ix+1]]];
        phits1[2] = hitsInGPU.idxs[mdsInGPU.outerHitIndices[segmentsInGPU.mdIndices[2*ix]]];
        phits1[3] = hitsInGPU.idxs[mdsInGPU.outerHitIndices[segmentsInGPU.mdIndices[2*ix+1]]];

        phits2[0] = hitsInGPU.idxs[mdsInGPU.anchorHitIndices[segmentsInGPU.mdIndices[2*jx]]];
        phits2[1] = hitsInGPU.idxs[mdsInGPU.anchorHitIndices[segmentsInGPU.mdIndices[2*jx+1]]];
        phits2[2] = hitsInGPU.idxs[mdsInGPU.outerHitIndices[segmentsInGPU.mdIndices[2*jx]]];
        phits2[3] = hitsInGPU.idxs[mdsInGPU.outerHitIndices[segmentsInGPU.mdIndices[2*jx+1]]];

        int npMatched = 0;

        for (int i = 0; i < 4; i++)
        {
            bool pmatched = false;
            if(phits1[i] == -1)
                continue;

            for (int j = 0; j < 4; j++)
            {
                if(phits2[j] == -1)
                    continue;

                if(phits1[i] == phits2[j])
                {
                    pmatched = true; break;
                }
            }
            if(pmatched)
                npMatched++;
        }
        return npMatched;
    };

    struct crossCleanpT3
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
                TAcc const & acc,
                struct SDL::modules& modulesInGPU,
                struct SDL::objectRanges& rangesInGPU,
                struct SDL::pixelTriplets& pixelTripletsInGPU,
                struct SDL::segments& segmentsInGPU,
                struct SDL::pixelQuintuplets& pixelQuintupletsInGPU) const
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;

            Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

            unsigned int nPixelTriplets = *pixelTripletsInGPU.nPixelTriplets;
            for(int pixelTripletIndex = globalThreadIdx[2]; pixelTripletIndex < nPixelTriplets; pixelTripletIndex += gridThreadExtent[2])
            {
                if(pixelTripletsInGPU.isDup[pixelTripletIndex])
                    continue;

                // Cross cleaning step
                float eta1 = __H2F(pixelTripletsInGPU.eta_pix[pixelTripletIndex]);
                float phi1 = __H2F(pixelTripletsInGPU.phi_pix[pixelTripletIndex]);

                int pixelModuleIndex = *modulesInGPU.nLowerModules;
                unsigned int prefix = rangesInGPU.segmentModuleIndices[pixelModuleIndex];

                unsigned int nPixelQuintuplets = *pixelQuintupletsInGPU.nPixelQuintuplets;
                for(int pixelQuintupletIndex = globalThreadIdx[1]; pixelQuintupletIndex < nPixelQuintuplets; pixelQuintupletIndex += gridThreadExtent[1])
                {
                    unsigned int pLS_jx = pixelQuintupletsInGPU.pixelIndices[pixelQuintupletIndex];
                    float eta2 = segmentsInGPU.eta[pLS_jx - prefix];
                    float phi2 = segmentsInGPU.phi[pLS_jx - prefix];
                    float dEta = alpaka::math::abs(acc, (eta1 - eta2));
                    float dPhi = SDL::calculate_dPhi(acc, phi1, phi2);

                    float dR2 = dEta*dEta + dPhi*dPhi;
                    if(dR2 < 1e-5f)
                        pixelTripletsInGPU.isDup[pixelTripletIndex] = true;
                }
            }
        }
    };

    struct crossCleanT5
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
                TAcc const & acc,
                struct SDL::modules& modulesInGPU,
                struct SDL::quintuplets& quintupletsInGPU,
                struct SDL::pixelQuintuplets& pixelQuintupletsInGPU,
                struct SDL::pixelTriplets& pixelTripletsInGPU,
                struct SDL::objectRanges& rangesInGPU) const
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;

            Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

            for(int innerInnerInnerLowerModuleArrayIndex = globalThreadIdx[0]; innerInnerInnerLowerModuleArrayIndex < *(modulesInGPU.nLowerModules); innerInnerInnerLowerModuleArrayIndex += gridThreadExtent[0])
            {
                if(rangesInGPU.quintupletModuleIndices[innerInnerInnerLowerModuleArrayIndex] == -1)
                    continue;

                unsigned int nQuints = quintupletsInGPU.nQuintuplets[innerInnerInnerLowerModuleArrayIndex];
                for(int innerObjectArrayIndex = globalThreadIdx[1]; innerObjectArrayIndex < nQuints; innerObjectArrayIndex += gridThreadExtent[1])
                {
                    int quintupletIndex = rangesInGPU.quintupletModuleIndices[innerInnerInnerLowerModuleArrayIndex] + innerObjectArrayIndex;

                    // Don't add duplicate T5s or T5s that are accounted in pT5s
                    if(quintupletsInGPU.isDup[quintupletIndex] or quintupletsInGPU.partOfPT5[quintupletIndex])
                        continue;
#ifdef Crossclean_T5
                    int loop_bound = *pixelQuintupletsInGPU.nPixelQuintuplets + *pixelTripletsInGPU.nPixelTriplets; 
                    // Cross cleaning step
                    float eta1 = __H2F(quintupletsInGPU.eta[quintupletIndex]);
                    float phi1 = __H2F(quintupletsInGPU.phi[quintupletIndex]);

                    for(unsigned int jx = globalThreadIdx[2]; jx < loop_bound; jx += gridThreadExtent[2])
                    {
                        float eta2, phi2;
                        if(jx < *pixelQuintupletsInGPU.nPixelQuintuplets)
                        {
                            eta2 = __H2F(pixelQuintupletsInGPU.eta[jx]);
                            phi2 = __H2F(pixelQuintupletsInGPU.phi[jx]);
                        }
                        else
                        {
                            eta2 = __H2F(pixelTripletsInGPU.eta[jx]);
                            phi2 = __H2F(pixelTripletsInGPU.phi[jx]);
                        }

                        float dEta = alpaka::math::abs(acc, eta1 - eta2);
                        float dPhi = SDL::calculate_dPhi(acc, phi1, phi2);

                        float dR2 = dEta*dEta + dPhi*dPhi;
                        if(dR2 < 1e-3f)
                            quintupletsInGPU.isDup[quintupletIndex] = true;
                    }
#endif
                }
            }
        }
    };

    // Using Matt's block for the outer loop and thread for inner loop trick here!
    // This will eliminate the need for another kernel just for adding the pLS, because we can __syncthreads()
    struct crossCleanpLS
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
                TAcc const & acc,
                struct SDL::modules& modulesInGPU,
                struct SDL::objectRanges& rangesInGPU,
                struct SDL::pixelTriplets& pixelTripletsInGPU,
                struct SDL::trackCandidates& trackCandidatesInGPU,
                struct SDL::segments& segmentsInGPU,
                struct SDL::miniDoublets& mdsInGPU,
                struct SDL::hits& hitsInGPU,
                struct SDL::quintuplets& quintupletsInGPU) const
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;

            Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

            int pixelModuleIndex = *modulesInGPU.nLowerModules;
            unsigned int nPixels = segmentsInGPU.nSegments[pixelModuleIndex];
            for(int pixelArrayIndex = globalThreadIdx[2]; pixelArrayIndex < nPixels; pixelArrayIndex += gridThreadExtent[2])
            {
                if(!segmentsInGPU.isQuad[pixelArrayIndex] || segmentsInGPU.isDup[pixelArrayIndex])
                    continue;

                float eta1 = segmentsInGPU.eta[pixelArrayIndex];
                float phi1 = segmentsInGPU.phi[pixelArrayIndex];
                unsigned int prefix = rangesInGPU.segmentModuleIndices[pixelModuleIndex];

                int nTrackCandidates = *(trackCandidatesInGPU.nTrackCandidates);
                for(int trackCandidateIndex = globalThreadIdx[1]; trackCandidateIndex < nTrackCandidates; trackCandidateIndex += gridThreadExtent[1])
                {
                    short type = trackCandidatesInGPU.trackCandidateType[trackCandidateIndex];
                    unsigned int innerTrackletIdx = trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex];
                    if(type == 4) // T5
                    {
                        unsigned int quintupletIndex = innerTrackletIdx; // T5 index
                        float eta2 = __H2F(quintupletsInGPU.eta[quintupletIndex]);
                        float phi2 = __H2F(quintupletsInGPU.phi[quintupletIndex]);
                        float dEta = alpaka::math::abs(acc, eta1 - eta2);
                        float dPhi = SDL::calculate_dPhi(acc, phi1, phi2);

                        float dR2 = dEta*dEta + dPhi*dPhi;
                        if(dR2 < 1e-3f)
                            segmentsInGPU.isDup[pixelArrayIndex] = true;
                    }
                    if(type == 5) // pT3
                    {
                        int pLSIndex = pixelTripletsInGPU.pixelSegmentIndices[innerTrackletIdx];
                        int npMatched = checkPixelHits(prefix+pixelArrayIndex,pLSIndex,mdsInGPU,segmentsInGPU,hitsInGPU);
                        if(npMatched > 0)
                            segmentsInGPU.isDup[pixelArrayIndex] = true;

                        int pT3Index = innerTrackletIdx;
                        float eta2 = __H2F(pixelTripletsInGPU.eta_pix[pT3Index]);
                        float phi2 = __H2F(pixelTripletsInGPU.phi_pix[pT3Index]);
                        float dEta = alpaka::math::abs(acc, eta1 - eta2);
                        float dPhi = SDL::calculate_dPhi(acc, phi1, phi2);

                        float dR2 = dEta*dEta + dPhi*dPhi;
                        if(dR2 < 0.000001f)
                            segmentsInGPU.isDup[pixelArrayIndex] = true;
                    }
                    if(type == 7) // pT5
                    {
                        unsigned int pLSIndex = innerTrackletIdx;
                        int npMatched = checkPixelHits(prefix+pixelArrayIndex,pLSIndex,mdsInGPU,segmentsInGPU,hitsInGPU);
                        if(npMatched >0) {segmentsInGPU.isDup[pixelArrayIndex] = true;}

                        float eta2 = segmentsInGPU.eta[pLSIndex - prefix];
                        float phi2 = segmentsInGPU.phi[pLSIndex - prefix];
                        float dEta = alpaka::math::abs(acc, eta1 - eta2);
                        float dPhi = SDL::calculate_dPhi(acc, phi1, phi2);

                        float dR2 = dEta*dEta + dPhi*dPhi;
                        if(dR2 < 0.000001f)
                            segmentsInGPU.isDup[pixelArrayIndex] = true;
                    }
                }
            }
        }
    };

    struct addpT3asTrackCandidatesInGPU
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
                TAcc const & acc,
                uint16_t nLowerModules,
                struct SDL::pixelTriplets& pixelTripletsInGPU,
                struct SDL::trackCandidates& trackCandidatesInGPU,
                struct SDL::segments& segmentsInGPU,
                struct SDL::objectRanges& rangesInGPU) const
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;

            Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

            unsigned int nPixelTriplets = *pixelTripletsInGPU.nPixelTriplets;
            unsigned int pLS_offset = rangesInGPU.segmentModuleIndices[nLowerModules];
            for(int pixelTripletIndex = globalThreadIdx[2]; pixelTripletIndex < nPixelTriplets; pixelTripletIndex += gridThreadExtent[2])
            {
                if((pixelTripletsInGPU.isDup[pixelTripletIndex]))
                    continue;

                unsigned int trackCandidateIdx = alpaka::atomicOp<alpaka::AtomicAdd>(acc, trackCandidatesInGPU.nTrackCandidates, 1);
                alpaka::atomicOp<alpaka::AtomicAdd>(acc, trackCandidatesInGPU.nTrackCandidatespT3, 1);

                float radius = 0.5f * (__H2F(pixelTripletsInGPU.pixelRadius[pixelTripletIndex]) + __H2F(pixelTripletsInGPU.tripletRadius[pixelTripletIndex]));
                unsigned int pT3PixelIndex =  pixelTripletsInGPU.pixelSegmentIndices[pixelTripletIndex];
                addTrackCandidateToMemory(trackCandidatesInGPU, 5/*track candidate type pT3=5*/, pixelTripletIndex, pixelTripletIndex, &pixelTripletsInGPU.logicalLayers[5 * pixelTripletIndex], &pixelTripletsInGPU.lowerModuleIndices[5 * pixelTripletIndex], &pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex], segmentsInGPU.seedIdx[pT3PixelIndex - pLS_offset], __H2F(pixelTripletsInGPU.centerX[pixelTripletIndex]), __H2F(pixelTripletsInGPU.centerY[pixelTripletIndex]),radius,trackCandidateIdx, pixelTripletIndex);
            }
        }
    };

    struct addT5asTrackCandidateInGPU
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
                TAcc const & acc,
                uint16_t nLowerModules,
                struct SDL::quintuplets& quintupletsInGPU,
                struct SDL::trackCandidates& trackCandidatesInGPU,
                struct SDL::objectRanges& rangesInGPU) const
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;

            Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

            for(int idx = globalThreadIdx[1]; idx < nLowerModules; idx += gridThreadExtent[1])
            {
                if(rangesInGPU.quintupletModuleIndices[idx] == -1)
                    continue;

                unsigned int nQuints = quintupletsInGPU.nQuintuplets[idx];
                for(int jdx = globalThreadIdx[2]; jdx < nQuints; jdx += gridThreadExtent[2])
                {
                    int quintupletIndex = rangesInGPU.quintupletModuleIndices[idx] + jdx;
<<<<<<< HEAD
                    if (quintupletsInGPU.isDup[quintupletIndex] or quintupletsInGPU.partOfPT5[quintupletIndex]) continue;
                    if (!(quintupletsInGPU.TightCutFlag[quintupletIndex])) continue;
=======
                    if(quintupletsInGPU.isDup[quintupletIndex] or quintupletsInGPU.partOfPT5[quintupletIndex])
                        continue;
>>>>>>> formatting fixes + fix break bug

                    unsigned int trackCandidateIdx = alpaka::atomicOp<alpaka::AtomicAdd>(acc, trackCandidatesInGPU.nTrackCandidates,1);
                    alpaka::atomicOp<alpaka::AtomicAdd>(acc, trackCandidatesInGPU.nTrackCandidatesT5,1);
                    addTrackCandidateToMemory(trackCandidatesInGPU, 4/*track candidate type T5=4*/, quintupletIndex, quintupletIndex, &quintupletsInGPU.logicalLayers[5 * quintupletIndex], &quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex], &quintupletsInGPU.hitIndices[10 * quintupletIndex], -1/*no pixel seed index for T5s*/, quintupletsInGPU.regressionG[quintupletIndex], quintupletsInGPU.regressionF[quintupletIndex], quintupletsInGPU.regressionRadius[quintupletIndex], trackCandidateIdx, quintupletIndex);
                }
            }
        }
    };

    struct addpLSasTrackCandidateInGPU
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
                TAcc const & acc,
                uint16_t nLowerModules,
                struct SDL::trackCandidates& trackCandidatesInGPU,
                struct SDL::segments& segmentsInGPU) const
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;

            Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

            unsigned int nPixels = segmentsInGPU.nSegments[nLowerModules];
            for(int pixelArrayIndex = globalThreadIdx[2]; pixelArrayIndex < nPixels; pixelArrayIndex += gridThreadExtent[2])
            {
                if((!segmentsInGPU.isQuad[pixelArrayIndex]) || (segmentsInGPU.isDup[pixelArrayIndex]))
                    continue;

                unsigned int trackCandidateIdx = alpaka::atomicOp<alpaka::AtomicAdd>(acc, trackCandidatesInGPU.nTrackCandidates, 1);
                alpaka::atomicOp<alpaka::AtomicAdd>(acc, trackCandidatesInGPU.nTrackCandidatespLS, 1);
                addpLSTrackCandidateToMemory(trackCandidatesInGPU, pixelArrayIndex, trackCandidateIdx, segmentsInGPU.pLSHitsIdxs[pixelArrayIndex], segmentsInGPU.seedIdx[pixelArrayIndex]);
            }
        }
    };

    struct addpT5asTrackCandidateInGPU
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(
                TAcc const & acc,
<<<<<<< HEAD
                uint16_t nLowerModules,
                struct SDL::pixelQuintuplets& pixelQuintupletsInGPU,
                struct SDL::trackCandidates& trackCandidatesInGPU,
                struct SDL::segments& segmentsInGPU,
                struct SDL::objectRanges& rangesInGPU) const
=======
                struct SDL::pixelQuintuplets& pixelQuintupletsInGPU,
                struct SDL::trackCandidates& trackCandidatesInGPU,
                struct SDL::quintuplets& quintupletsInGPU) const
>>>>>>> formatting fixes + fix break bug
        {
            using Dim = alpaka::Dim<TAcc>;
            using Idx = alpaka::Idx<TAcc>;
            using Vec = alpaka::Vec<Dim, Idx>;

            Vec const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            Vec const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

            int nPixelQuintuplets = *pixelQuintupletsInGPU.nPixelQuintuplets;
            unsigned int pLS_offset = rangesInGPU.segmentModuleIndices[nLowerModules];
            for(int pixelQuintupletIndex = globalThreadIdx[2]; pixelQuintupletIndex < nPixelQuintuplets; pixelQuintupletIndex += gridThreadExtent[2])
            {
                if(pixelQuintupletsInGPU.isDup[pixelQuintupletIndex])
                    continue;

                int trackCandidateIdx = alpaka::atomicOp<alpaka::AtomicAdd>(acc, trackCandidatesInGPU.nTrackCandidates, 1);
                alpaka::atomicOp<alpaka::AtomicAdd>(acc, trackCandidatesInGPU.nTrackCandidatespT5,1);

                float radius = 0.5f*(__H2F(pixelQuintupletsInGPU.pixelRadius[pixelQuintupletIndex]) + __H2F(pixelQuintupletsInGPU.quintupletRadius[pixelQuintupletIndex]));
<<<<<<< HEAD
                unsigned int pT5PixelIndex =  pixelQuintupletsInGPU.pixelIndices[pixelQuintupletIndex];
                addTrackCandidateToMemory(trackCandidatesInGPU, 7/*track candidate type pT5=7*/, pT5PixelIndex, pixelQuintupletsInGPU.T5Indices[pixelQuintupletIndex], &pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex], &pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex], &pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex], segmentsInGPU.seedIdx[pT5PixelIndex - pLS_offset], __H2F(pixelQuintupletsInGPU.centerX[pixelQuintupletIndex]), __H2F(pixelQuintupletsInGPU.centerY[pixelQuintupletIndex]),radius , trackCandidateIdx, pixelQuintupletIndex);
=======
                addTrackCandidateToMemory(trackCandidatesInGPU, 7/*track candidate type pT5=7*/, pixelQuintupletsInGPU.pixelIndices[pixelQuintupletIndex], pixelQuintupletsInGPU.T5Indices[pixelQuintupletIndex], &pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex], &pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex], &pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex], __H2F(pixelQuintupletsInGPU.centerX[pixelQuintupletIndex]), __H2F(pixelQuintupletsInGPU.centerY[pixelQuintupletIndex]), radius, trackCandidateIdx, pixelQuintupletIndex);
>>>>>>> formatting fixes + fix break bug
            }
        }
    };
}
#endif
