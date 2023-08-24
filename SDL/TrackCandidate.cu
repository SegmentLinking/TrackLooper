#include "TrackCandidate.cuh"

#include "allocate.h"


void SDL::trackCandidates::resetMemory(unsigned int maxTrackCandidates,cudaStream_t stream)
{
    cudaMemsetAsync(trackCandidateType,0, maxTrackCandidates * sizeof(short),stream);
    cudaMemsetAsync(directObjectIndices, 0, maxTrackCandidates * sizeof(unsigned int),stream);
    cudaMemsetAsync(objectIndices, 0,2 * maxTrackCandidates * sizeof(unsigned int),stream);
    cudaMemsetAsync(nTrackCandidates, 0,sizeof(unsigned int),stream);
    cudaMemsetAsync(nTrackCandidatespT3, 0,sizeof(unsigned int),stream);
    cudaMemsetAsync(nTrackCandidatesT5, 0,sizeof(unsigned int),stream);
    cudaMemsetAsync(nTrackCandidatespT5,0, sizeof(unsigned int),stream);
    cudaMemsetAsync(nTrackCandidatespLS, 0,sizeof(unsigned int),stream);

    cudaMemsetAsync(logicalLayers, 0, 7 * maxTrackCandidates * sizeof(uint8_t), stream);
    cudaMemsetAsync(lowerModuleIndices, 0, 7 * maxTrackCandidates * sizeof(uint16_t), stream);
    cudaMemsetAsync(hitIndices, 0, 14 * maxTrackCandidates * sizeof(unsigned int), stream);
    cudaMemsetAsync(pixelSeedIndex, 0, maxTrackCandidates * sizeof(int), stream);
    cudaMemsetAsync(centerX, 0, maxTrackCandidates * sizeof(FPX), stream);
    cudaMemsetAsync(centerY, 0, maxTrackCandidates * sizeof(FPX), stream);
    cudaMemsetAsync(radius , 0, maxTrackCandidates * sizeof(FPX), stream);
}

void SDL::createTrackCandidatesInExplicitMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int maxTrackCandidates,cudaStream_t stream)
{
#ifdef CACHE_ALLOC
    int dev;
    cudaGetDevice(&dev);
    trackCandidatesInGPU.trackCandidateType = (short*)cms::cuda::allocate_device(dev,maxTrackCandidates * sizeof(short),stream);
    trackCandidatesInGPU.directObjectIndices = (unsigned int*)cms::cuda::allocate_device(dev,maxTrackCandidates * sizeof(unsigned int),stream);
    trackCandidatesInGPU.objectIndices = (unsigned int*)cms::cuda::allocate_device(dev,maxTrackCandidates * 2*sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidates= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespT3= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatesT5= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespT5= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespLS= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);

    trackCandidatesInGPU.logicalLayers = (uint8_t*)cms::cuda::allocate_device(dev, 7 * maxTrackCandidates * sizeof(uint8_t), stream);
    trackCandidatesInGPU.lowerModuleIndices = (uint16_t*)cms::cuda::allocate_device(dev, 7 * maxTrackCandidates * sizeof(uint16_t), stream);
    trackCandidatesInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_device(dev, 14 * maxTrackCandidates * sizeof(unsigned int), stream);
    trackCandidatesInGPU.pixelSeedIndex = (int*)cms::cuda::allocate_device(dev, maxTrackCandidates * sizeof(int), stream);
    trackCandidatesInGPU.centerX = (FPX*)cms::cuda::allocate_device(dev, maxTrackCandidates * sizeof(FPX), stream);
    trackCandidatesInGPU.centerY = (FPX*)cms::cuda::allocate_device(dev, maxTrackCandidates * sizeof(FPX), stream);
    trackCandidatesInGPU.radius  = (FPX*)cms::cuda::allocate_device(dev, maxTrackCandidates * sizeof(FPX), stream);

#else
    cudaMalloc(&trackCandidatesInGPU.trackCandidateType, maxTrackCandidates * sizeof(short));
    cudaMalloc(&trackCandidatesInGPU.directObjectIndices, maxTrackCandidates * sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.objectIndices, 2 * maxTrackCandidates * sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidates, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatespT3, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatesT5, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatespT5, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatespLS, sizeof(unsigned int));

    cudaMalloc(&trackCandidatesInGPU.logicalLayers, 7 * maxTrackCandidates * sizeof(uint8_t));
    cudaMalloc(&trackCandidatesInGPU.lowerModuleIndices, 7 * maxTrackCandidates * sizeof(uint16_t));
    cudaMalloc(&trackCandidatesInGPU.hitIndices, 14 * maxTrackCandidates * sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.pixelSeedIndex, maxTrackCandidates * sizeof(int));
    cudaMalloc(&trackCandidatesInGPU.centerX, maxTrackCandidates * sizeof(FPX));
    cudaMalloc(&trackCandidatesInGPU.centerY, maxTrackCandidates * sizeof(FPX));
    cudaMalloc(&trackCandidatesInGPU.radius , maxTrackCandidates * sizeof(FPX));
#endif
    cudaMemsetAsync(trackCandidatesInGPU.nTrackCandidates,0, sizeof(unsigned int), stream);
    cudaMemsetAsync(trackCandidatesInGPU.nTrackCandidatesT5,0, sizeof(unsigned int), stream);
    cudaMemsetAsync(trackCandidatesInGPU.nTrackCandidatespT3,0, sizeof(unsigned int), stream);
    cudaMemsetAsync(trackCandidatesInGPU.nTrackCandidatespT5,0, sizeof(unsigned int), stream);
    cudaMemsetAsync(trackCandidatesInGPU.nTrackCandidatespLS,0, sizeof(unsigned int), stream);
    cudaMemsetAsync(trackCandidatesInGPU.logicalLayers, 0, 7 * maxTrackCandidates * sizeof(uint8_t), stream);
    cudaMemsetAsync(trackCandidatesInGPU.lowerModuleIndices, 0, 7 * maxTrackCandidates * sizeof(uint16_t), stream);
    cudaMemsetAsync(trackCandidatesInGPU.hitIndices, 0, 14 * maxTrackCandidates * sizeof(unsigned int), stream);
    cudaMemsetAsync(trackCandidatesInGPU.pixelSeedIndex, 0, maxTrackCandidates * sizeof(int), stream);
    cudaStreamSynchronize(stream);
}

__device__ void SDL::addpLSTrackCandidateToMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int trackletIndex, unsigned int trackCandidateIndex, uint4 hitIndices, int pixelSeedIndex)
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
}

__device__ void SDL::addTrackCandidateToMemory(struct trackCandidates& trackCandidatesInGPU, short trackCandidateType, unsigned int innerTrackletIndex, unsigned int outerTrackletIndex, uint8_t* logicalLayerIndices, uint16_t* lowerModuleIndices, unsigned int* hitIndices, int pixelSeedIndex, float centerX, float centerY, float radius, unsigned int trackCandidateIndex, unsigned int directObjectIndex)
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
}

SDL::trackCandidates::trackCandidates()
{
    trackCandidateType = nullptr;
    directObjectIndices = nullptr;
    objectIndices = nullptr;
    nTrackCandidates = nullptr;
    nTrackCandidatesT5 = nullptr;
    nTrackCandidatespT3 = nullptr;
    nTrackCandidatespT5 = nullptr;
    nTrackCandidatespLS = nullptr;

    logicalLayers = nullptr;
    hitIndices = nullptr;
    pixelSeedIndex = nullptr;
    lowerModuleIndices = nullptr;
    centerX = nullptr;
    centerY = nullptr;
    radius = nullptr;
}

SDL::trackCandidates::~trackCandidates()
{
}

void SDL::trackCandidates::freeMemoryCache()
{
    int dev;
    cudaGetDevice(&dev);
    //FIXME
    //cudaFree(trackCandidateType);
    cms::cuda::free_device(dev,directObjectIndices);
    cms::cuda::free_device(dev,objectIndices);
    cms::cuda::free_device(dev,trackCandidateType);
    cms::cuda::free_device(dev,nTrackCandidates);
    cms::cuda::free_device(dev,nTrackCandidatespT3);
    cms::cuda::free_device(dev,nTrackCandidatesT5);
    cms::cuda::free_device(dev,nTrackCandidatespT5);
    cms::cuda::free_device(dev,nTrackCandidatespLS);

    cms::cuda::free_device(dev, logicalLayers);
    cms::cuda::free_device(dev, hitIndices);
    cms::cuda::free_device(dev, pixelSeedIndex);
    cms::cuda::free_device(dev, lowerModuleIndices);
    cms::cuda::free_device(dev, centerX);
    cms::cuda::free_device(dev, centerY);
    cms::cuda::free_device(dev, radius);
}
void SDL::trackCandidates::freeMemory(cudaStream_t stream)
{
    cudaFree(trackCandidateType);
    cudaFree(directObjectIndices);
    cudaFree(objectIndices);
    cudaFree(nTrackCandidates);
    cudaFree(nTrackCandidatespT3);
    cudaFree(nTrackCandidatesT5);
    cudaFree(nTrackCandidatespT5);
    cudaFree(nTrackCandidatespLS);

    cudaFree(logicalLayers);
    cudaFree(hitIndices);
    cudaFree(pixelSeedIndex);
    cudaFree(lowerModuleIndices);
    cudaFree(centerX);
    cudaFree(centerY);
    cudaFree(radius);
    
    cudaStreamSynchronize(stream);
}

__global__ void SDL::addpT5asTrackCandidateInGPU(uint16_t nLowerModules, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::segments& segmentsInGPU, struct SDL::objectRanges& rangesInGPU)
{
    unsigned int nPixelQuintuplets = *pixelQuintupletsInGPU.nPixelQuintuplets;
    unsigned int pLS_offset = rangesInGPU.segmentModuleIndices[nLowerModules];

    for(int pixelQuintupletIndex = blockIdx.x * blockDim.x + threadIdx.x; pixelQuintupletIndex < nPixelQuintuplets; pixelQuintupletIndex += blockDim.x*gridDim.x)
    {
        if(pixelQuintupletsInGPU.isDup[pixelQuintupletIndex])
        {
            continue;
        }
        unsigned int trackCandidateIdx = atomicAdd(trackCandidatesInGPU.nTrackCandidates,1);
        atomicAdd(trackCandidatesInGPU.nTrackCandidatespT5,1);


        float radius = 0.5f*(__H2F(pixelQuintupletsInGPU.pixelRadius[pixelQuintupletIndex]) + __H2F(pixelQuintupletsInGPU.quintupletRadius[pixelQuintupletIndex]));
        unsigned int pT5PixelIndex =  pixelQuintupletsInGPU.pixelIndices[pixelQuintupletIndex];
        addTrackCandidateToMemory(trackCandidatesInGPU, 7/*track candidate type pT5=7*/, pT5PixelIndex, pixelQuintupletsInGPU.T5Indices[pixelQuintupletIndex], &pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex], &pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex], &pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex], segmentsInGPU.seedIdx[pT5PixelIndex - pLS_offset], __H2F(pixelQuintupletsInGPU.centerX[pixelQuintupletIndex]),
                            __H2F(pixelQuintupletsInGPU.centerY[pixelQuintupletIndex]),radius , trackCandidateIdx, pixelQuintupletIndex);
    }
}

__global__ void SDL::crossCleanpT3(struct SDL::modules& modulesInGPU, struct SDL::objectRanges& rangesInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU)
{
    unsigned int nPixelTriplets = *pixelTripletsInGPU.nPixelTriplets;
    for(int pixelTripletIndex = blockIdx.x * blockDim.x + threadIdx.x; pixelTripletIndex < nPixelTriplets; pixelTripletIndex += blockDim.x*gridDim.x)
    {
        if(pixelTripletsInGPU.isDup[pixelTripletIndex]) continue;
        //cross cleaning step
        float eta1 = __H2F(pixelTripletsInGPU.eta_pix[pixelTripletIndex]);
        float phi1 = __H2F(pixelTripletsInGPU.phi_pix[pixelTripletIndex]);

        int pixelModuleIndex = *modulesInGPU.nLowerModules;
        unsigned int prefix = rangesInGPU.segmentModuleIndices[pixelModuleIndex];

        unsigned int nPixelQuintuplets = *pixelQuintupletsInGPU.nPixelQuintuplets;
        for(int pixelQuintupletIndex = blockIdx.y * blockDim.y + threadIdx.y; pixelQuintupletIndex < nPixelQuintuplets; pixelQuintupletIndex += blockDim.y*gridDim.y)
        {
            unsigned int pLS_jx = pixelQuintupletsInGPU.pixelIndices[pixelQuintupletIndex];
            float eta2 = segmentsInGPU.eta[pLS_jx - prefix];
            float phi2 = segmentsInGPU.phi[pLS_jx - prefix];
            float dEta = abs(eta1-eta2);
            float dPhi = abs(phi1-phi2);
            if(dPhi > float(M_PI)){dPhi = dPhi - 2*float(M_PI);}
            float dR2 = dEta*dEta + dPhi*dPhi;
            if(dR2 < 1e-5f) pixelTripletsInGPU.isDup[pixelTripletIndex] = true;
        }
    }
}

__global__ void SDL::crossCleanT5(struct SDL::modules& modulesInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU,struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::objectRanges& rangesInGPU)
{
    int stepx = blockDim.x*gridDim.x;
    int stepy = blockDim.y*gridDim.y;
    int stepz = blockDim.z*gridDim.z;
    for(int innerInnerInnerLowerModuleArrayIndex = blockIdx.z * blockDim.z + threadIdx.z; innerInnerInnerLowerModuleArrayIndex < *(modulesInGPU.nLowerModules); innerInnerInnerLowerModuleArrayIndex+=stepz)
    {
        if(rangesInGPU.quintupletModuleIndices[innerInnerInnerLowerModuleArrayIndex] == -1) continue;
        unsigned int nQuints = quintupletsInGPU.nQuintuplets[innerInnerInnerLowerModuleArrayIndex];
        for(int innerObjectArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;innerObjectArrayIndex < nQuints;innerObjectArrayIndex+=stepy)
        {
            int quintupletIndex = rangesInGPU.quintupletModuleIndices[innerInnerInnerLowerModuleArrayIndex] + innerObjectArrayIndex;

            //don't add duplicate T5s or T5s that are accounted in pT5s
            if(quintupletsInGPU.isDup[quintupletIndex] or quintupletsInGPU.partOfPT5[quintupletIndex])
            {
                continue;//return;
            }
#ifdef Crossclean_T5
            int loop_bound = *pixelQuintupletsInGPU.nPixelQuintuplets + *pixelTripletsInGPU.nPixelTriplets; 
            //cross cleaning step
            float eta1 = __H2F(quintupletsInGPU.eta[quintupletIndex]);
            float phi1 = __H2F(quintupletsInGPU.phi[quintupletIndex]);

            for (unsigned int jx=blockIdx.x * blockDim.x + threadIdx.x; jx<loop_bound; jx+=stepx)
            {
                float eta2, phi2;
                if(jx < *pixelQuintupletsInGPU.nPixelQuintuplets)
                {
                    eta2 = __H2F(pixelQuintupletsInGPU.eta[jx]);
                    phi2 = __H2F(pixelQuintupletsInGPU.phi[jx]);
                }
                else
                {
                    eta2 = __H2F(pixelTripletsInGPU.eta[jx - *pixelQuintupletsInGPU.nPixelQuintuplets]);
                    phi2 = __H2F(pixelTripletsInGPU.phi[jx - *pixelQuintupletsInGPU.nPixelQuintuplets]);
                }

                float dEta = abs(eta1-eta2);
                float dPhi = abs(phi1-phi2);
                if(dPhi > float(M_PI)){dPhi = dPhi - 2*float(M_PI);}
                float dR2 = dEta*dEta + dPhi*dPhi;
                if(dR2 < 1e-3f) {quintupletsInGPU.isDup[quintupletIndex] = true;}//return;

            }
#endif
        }
    }
}

//Using Matt's block for the outer loop and thread for inner loop trick here!
//This will eliminate the need for another kernel just for adding the pLS, because we can __syncthreads()
__global__ void SDL::crossCleanpLS(struct SDL::modules& modulesInGPU, struct SDL::objectRanges& rangesInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU,struct SDL::segments& segmentsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::hits& hitsInGPU, struct SDL::quintuplets& quintupletsInGPU)
{
    int pixelModuleIndex = *modulesInGPU.nLowerModules;
    unsigned int nPixels = segmentsInGPU.nSegments[pixelModuleIndex];
    for(int pixelArrayIndex = blockIdx.x * blockDim.x + threadIdx.x; pixelArrayIndex < nPixels; pixelArrayIndex+=blockDim.x*gridDim.x)
    {
        if((!segmentsInGPU.isQuad[pixelArrayIndex]) || (segmentsInGPU.isDup[pixelArrayIndex])) {continue;}

        float eta1 = segmentsInGPU.eta[pixelArrayIndex];
        float phi1 = segmentsInGPU.phi[pixelArrayIndex];
        unsigned int prefix = rangesInGPU.segmentModuleIndices[pixelModuleIndex];

        unsigned int nTrackCandidates = *(trackCandidatesInGPU.nTrackCandidates);
        for(int trackCandidateIndex = blockIdx.y * blockDim.y + threadIdx.y; trackCandidateIndex < nTrackCandidates; trackCandidateIndex+=blockDim.y*gridDim.y)
        {
            short type = trackCandidatesInGPU.trackCandidateType[trackCandidateIndex];
            unsigned int innerTrackletIdx = trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex];
            if(type == 4) //T5
            {
                unsigned int quintupletIndex = innerTrackletIdx;//trackCandidatesInGPU.objectIndices[2*jx];//T5 index
                float eta2 = __H2F(quintupletsInGPU.eta[quintupletIndex]);
                float phi2 = __H2F(quintupletsInGPU.phi[quintupletIndex]);
                float dEta = abs(eta1-eta2);
                float dPhi = abs(phi1-phi2);
                if(dPhi > float(M_PI)){dPhi = dPhi - 2*float(M_PI);}
                float dR2 = dEta*dEta + dPhi*dPhi;
                if(dR2 < 1e-3f) {segmentsInGPU.isDup[pixelArrayIndex] = true;}
            }
            if(type == 5)  //pT3
            {
                int pLSIndex = pixelTripletsInGPU.pixelSegmentIndices[innerTrackletIdx];
                int npMatched = checkPixelHits(prefix+pixelArrayIndex,pLSIndex,mdsInGPU,segmentsInGPU,hitsInGPU);
                if(npMatched >0) {segmentsInGPU.isDup[pixelArrayIndex] = true;}

                int pT3Index = innerTrackletIdx;
                float eta2 = __H2F(pixelTripletsInGPU.eta_pix[pT3Index]);
                float phi2 = __H2F(pixelTripletsInGPU.phi_pix[pT3Index]);
                float dEta = abs(eta1-eta2);
                float dPhi = abs(phi1-phi2);
                if(dPhi > float(M_PI)){dPhi = dPhi - 2*float(M_PI);}
                float dR2 = dEta*dEta + dPhi*dPhi;
                if(dR2 < 0.000001f) {segmentsInGPU.isDup[pixelArrayIndex] = true;}
            }
            if(type == 7) //pT5
            {
                unsigned int pLSIndex = innerTrackletIdx;
                int npMatched = checkPixelHits(prefix+pixelArrayIndex,pLSIndex,mdsInGPU,segmentsInGPU,hitsInGPU);
                if(npMatched >0) {segmentsInGPU.isDup[pixelArrayIndex] = true;}

                float eta2 = segmentsInGPU.eta[pLSIndex - prefix];
                float phi2 = segmentsInGPU.phi[pLSIndex - prefix];
                float dEta = abs(eta1-eta2);
                float dPhi = abs(phi1-phi2);
                if(dPhi > float(M_PI)){dPhi = dPhi - 2*float(M_PI);}
                float dR2 = dEta*dEta + dPhi*dPhi;
                if(dR2 < 0.000001f) {segmentsInGPU.isDup[pixelArrayIndex] = true;}
            }
        }
    }
}

__global__ void SDL::addpT3asTrackCandidatesInGPU(uint16_t nLowerModules, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::segments& segmentsInGPU, struct SDL::objectRanges& rangesInGPU)
{
    unsigned int nPixelTriplets = *pixelTripletsInGPU.nPixelTriplets;
    unsigned int pLS_offset = rangesInGPU.segmentModuleIndices[nLowerModules];

    for(int pixelTripletIndex = blockIdx.x * blockDim.x + threadIdx.x; pixelTripletIndex < nPixelTriplets; pixelTripletIndex += blockDim.x*gridDim.x)
    {
        if((pixelTripletsInGPU.isDup[pixelTripletIndex])) continue;//return;
        unsigned int trackCandidateIdx = atomicAdd(trackCandidatesInGPU.nTrackCandidates,1);
        atomicAdd(trackCandidatesInGPU.nTrackCandidatespT3,1);
    
        float radius = 0.5f * (__H2F(pixelTripletsInGPU.pixelRadius[pixelTripletIndex]) + __H2F(pixelTripletsInGPU.tripletRadius[pixelTripletIndex]));
        unsigned int pT3PixelIndex =  pixelTripletsInGPU.pixelSegmentIndices[pixelTripletIndex];
        addTrackCandidateToMemory(trackCandidatesInGPU, 5/*track candidate type pT3=5*/, pixelTripletIndex, pixelTripletIndex, &pixelTripletsInGPU.logicalLayers[5 * pixelTripletIndex], &pixelTripletsInGPU.lowerModuleIndices[5 * pixelTripletIndex], &pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex], segmentsInGPU.seedIdx[pT3PixelIndex - pLS_offset], __H2F(pixelTripletsInGPU.centerX[pixelTripletIndex]), __H2F(pixelTripletsInGPU.centerY[pixelTripletIndex]),radius,trackCandidateIdx, pixelTripletIndex);
    }    

}

__global__ void SDL::addT5asTrackCandidateInGPU(uint16_t nLowerModules, struct SDL::quintuplets& quintupletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::objectRanges& rangesInGPU)
{
    for(uint16_t idx = blockIdx.y * blockDim.y + threadIdx.y; idx < nLowerModules; idx+= gridDim.y * blockDim.y)
    {
        if(rangesInGPU.quintupletModuleIndices[idx] == -1) continue;
        unsigned int nQuints = quintupletsInGPU.nQuintuplets[idx];
        for(unsigned int jdx = blockIdx.x * blockDim.x + threadIdx.x; jdx < nQuints; jdx += blockDim.x * gridDim.x)
        {
            int quintupletIndex = rangesInGPU.quintupletModuleIndices[idx] + jdx;

            if (quintupletsInGPU.isDup[quintupletIndex] or quintupletsInGPU.partOfPT5[quintupletIndex]) continue;

            if (!(quintupletsInGPU.TightCutFlag[quintupletIndex])) continue;

            unsigned int trackCandidateIdx = atomicAdd(trackCandidatesInGPU.nTrackCandidates,1);
            atomicAdd(trackCandidatesInGPU.nTrackCandidatesT5,1);
    
            addTrackCandidateToMemory(trackCandidatesInGPU, 4/*track candidate type T5=4*/, quintupletIndex, quintupletIndex, &quintupletsInGPU.logicalLayers[5 * quintupletIndex], &quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex], &quintupletsInGPU.hitIndices[10 * quintupletIndex], -1/*no pixel seed index for T5s*/, quintupletsInGPU.regressionG[quintupletIndex], quintupletsInGPU.regressionF[quintupletIndex], quintupletsInGPU.regressionRadius[quintupletIndex], trackCandidateIdx, quintupletIndex);
        } 
    }
}

__global__ void SDL::addpLSasTrackCandidateInGPU(uint16_t nLowerModules, struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::segments& segmentsInGPU)
{
    //int pixelArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    unsigned int nPixels = segmentsInGPU.nSegments[nLowerModules];
    for(int pixelArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;pixelArrayIndex < nPixels;  pixelArrayIndex +=step)
    {
        if((!segmentsInGPU.isQuad[pixelArrayIndex]) || (segmentsInGPU.isDup[pixelArrayIndex]))
        {
            continue;//return;
        }

        unsigned int trackCandidateIdx = atomicAdd(trackCandidatesInGPU.nTrackCandidates,1);
        atomicAdd(trackCandidatesInGPU.nTrackCandidatespLS,1);
        addpLSTrackCandidateToMemory(trackCandidatesInGPU, pixelArrayIndex, trackCandidateIdx, segmentsInGPU.pLSHitsIdxs[pixelArrayIndex], segmentsInGPU.seedIdx[pixelArrayIndex]);
    }
}


__device__ int SDL::checkPixelHits(unsigned int ix, unsigned int jx,struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::hits& hitsInGPU)
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

    for (int i =0; i<4;i++)
    {
        bool pmatched = false;
        if(phits1[i] == -1){continue;}
        for (int j =0; j<4; j++)
        {
            if(phits2[j] == -1){continue;}
            if(phits1[i] == phits2[j]){pmatched = true; break;}
        }
        if(pmatched){npMatched++;}
    }
    return npMatched;
}
