# include "Hit.cuh"
# include "allocate.h"
#ifdef __CUDACC__
#define CUDA_CONST_VAR __device__
#endif

SDL::hits::hits()
{
    nHits = nullptr;
//    n2SHits = nullptr;
    xs = nullptr;
    ys = nullptr;
    zs = nullptr;
    moduleIndices = nullptr;
    rts = nullptr;
    phis = nullptr;
    etas = nullptr;
    highEdgeXs = nullptr;
    highEdgeYs = nullptr;
    lowEdgeXs = nullptr;
    lowEdgeYs = nullptr;
}

SDL::hits::~hits()
{
}
//FIXME:New array!
void SDL::createHitsInUnifiedMemory(struct hits& hitsInGPU,unsigned int nMaxHits,unsigned int nMax2SHits,cudaStream_t stream)
{
#ifdef CACHE_ALLOC
//    cudaStream_t stream=0;
    hitsInGPU.xs = (float*)cms::cuda::allocate_managed(nMaxHits*sizeof(float),stream);
    hitsInGPU.ys = (float*)cms::cuda::allocate_managed(nMaxHits*sizeof(float),stream);
    hitsInGPU.zs = (float*)cms::cuda::allocate_managed(nMaxHits*sizeof(float),stream);

    hitsInGPU.rts = (float*)cms::cuda::allocate_managed(nMaxHits*sizeof(float),stream);
    hitsInGPU.phis = (float*)cms::cuda::allocate_managed(nMaxHits*sizeof(float),stream);
    hitsInGPU.etas = (float*)cms::cuda::allocate_managed(nMaxHits*sizeof(float),stream);

    hitsInGPU.moduleIndices = (uint16_t*)cms::cuda::allocate_managed(nMaxHits*sizeof(uint16_t),stream);
    hitsInGPU.idxs = (unsigned int*)cms::cuda::allocate_managed(nMaxHits*sizeof(unsigned int),stream);

    hitsInGPU.highEdgeXs = (float*)cms::cuda::allocate_managed(nMaxHits*sizeof(float),stream);
    hitsInGPU.highEdgeYs = (float*)cms::cuda::allocate_managed(nMaxHits*sizeof(float),stream);
    hitsInGPU.lowEdgeXs = (float*)cms::cuda::allocate_managed(nMaxHits*sizeof(float),stream);
    hitsInGPU.lowEdgeYs = (float*)cms::cuda::allocate_managed(nMaxHits*sizeof(float),stream);

    hitsInGPU.nHits = (unsigned int*)cms::cuda::allocate_managed(sizeof(unsigned int),stream);
#else
    //nMaxHits and nMax2SHits are the maximum possible numbers
    cudaMallocManaged(&hitsInGPU.xs, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.ys, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.zs, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.moduleIndices, nMaxHits * sizeof(uint16_t));
    //TODO:This dude (idxs) is not used in the GPU at all. It is only used for simhit matching to make efficiency plots
    //We can even skip this one later
    cudaMallocManaged(&hitsInGPU.idxs, nMaxHits * sizeof(unsigned int));

    cudaMallocManaged(&hitsInGPU.rts, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.phis, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.etas, nMaxHits * sizeof(float));

    cudaMallocManaged(&hitsInGPU.highEdgeXs, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.highEdgeYs, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.lowEdgeXs, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.lowEdgeYs, nMaxHits * sizeof(float));

    //counters
    cudaMallocManaged(&hitsInGPU.nHits, sizeof(unsigned int));
#endif
    *hitsInGPU.nHits = 0;
}
void SDL::createHitsInExplicitMemory(struct hits& hitsInGPU, unsigned int nMaxHits,cudaStream_t stream)
{
#ifdef CACHE_ALLOC
 //   cudaStream_t stream=0;
    int dev;
    cudaGetDevice(&dev);
    hitsInGPU.xs = (float*)cms::cuda::allocate_device(dev,nMaxHits*sizeof(float),stream);
    hitsInGPU.ys = (float*)cms::cuda::allocate_device(dev,nMaxHits*sizeof(float),stream);
    hitsInGPU.zs = (float*)cms::cuda::allocate_device(dev,nMaxHits*sizeof(float),stream);

    hitsInGPU.rts = (float*)cms::cuda::allocate_device(dev,nMaxHits*sizeof(float),stream);
    hitsInGPU.phis = (float*)cms::cuda::allocate_device(dev,nMaxHits*sizeof(float),stream);
    hitsInGPU.etas = (float*)cms::cuda::allocate_device(dev,nMaxHits*sizeof(float),stream);

    hitsInGPU.moduleIndices = (uint16_t*)cms::cuda::allocate_device(dev,nMaxHits*sizeof(uint16_t),stream);
    hitsInGPU.idxs = (unsigned int*)cms::cuda::allocate_device(dev,nMaxHits*sizeof(unsigned int),stream);

    hitsInGPU.highEdgeXs = (float*)cms::cuda::allocate_device(dev,nMaxHits*sizeof(float),stream);
    hitsInGPU.highEdgeYs = (float*)cms::cuda::allocate_device(dev,nMaxHits*sizeof(float),stream);
    hitsInGPU.lowEdgeXs = (float*)cms::cuda::allocate_device(dev,nMaxHits*sizeof(float),stream);
    hitsInGPU.lowEdgeYs = (float*)cms::cuda::allocate_device(dev,nMaxHits*sizeof(float),stream);

    hitsInGPU.nHits = (unsigned int*)cms::cuda::allocate_device(dev,sizeof(unsigned int),stream);
#else
    cudaMalloc(&hitsInGPU.xs, nMaxHits * sizeof(float));
    cudaMalloc(&hitsInGPU.ys, nMaxHits * sizeof(float));
    cudaMalloc(&hitsInGPU.zs, nMaxHits * sizeof(float));

    cudaMalloc(&hitsInGPU.moduleIndices, nMaxHits * sizeof(uint16_t));
    cudaMalloc(&hitsInGPU.idxs, nMaxHits * sizeof(unsigned int));

    cudaMalloc(&hitsInGPU.rts, nMaxHits * sizeof(float));
    cudaMalloc(&hitsInGPU.phis, nMaxHits * sizeof(float));
    cudaMalloc(&hitsInGPU.etas, nMaxHits * sizeof(float));

    cudaMalloc(&hitsInGPU.highEdgeXs, nMaxHits * sizeof(float));
    cudaMalloc(&hitsInGPU.highEdgeYs, nMaxHits * sizeof(float));
    cudaMalloc(&hitsInGPU.lowEdgeXs, nMaxHits * sizeof(float));
    cudaMalloc(&hitsInGPU.lowEdgeYs, nMaxHits * sizeof(float));

    //counters
    cudaMalloc(&hitsInGPU.nHits, sizeof(unsigned int));
#endif
    cudaMemsetAsync(hitsInGPU.nHits,0,sizeof(unsigned int),stream);
    cudaStreamSynchronize(stream);
}

__global__ void SDL::addHitToMemoryKernel(struct hits& hitsInGPU, struct modules& modulesInGPU,const float* x,const  float* y,const  float* z, const uint16_t* moduleIndex,const float* phis, const int loopsize)
{
  for (unsigned int ihit = blockIdx.x*blockDim.x + threadIdx.x; ihit <loopsize; ihit += blockDim.x*gridDim.x)
  //if(ihit < loopsize)
  {
      unsigned int idx = ihit;//*(hitsInGPU.nHits);

      hitsInGPU.xs[idx] = x[ihit];
      hitsInGPU.ys[idx] = y[ihit];
      hitsInGPU.zs[idx] = z[ihit];
      hitsInGPU.rts[idx] = sqrt(x[ihit]*x[ihit] + y[ihit]*y[ihit]);
      hitsInGPU.phis[idx] = phi(x[ihit],y[ihit],z[ihit]);
      hitsInGPU.moduleIndices[idx] = moduleIndex[ihit];
      hitsInGPU.idxs[idx] = ihit;
  }
}

__device__ void SDL::getEdgeHitsK(float phi,float x, float y, float& xhigh, float& yhigh, float& xlow, float& ylow)
{
    xhigh = x + 2.5 * cos(phi);
    yhigh = y + 2.5 * sin(phi);
    xlow = x - 2.5 * cos(phi);
    ylow = y - 2.5 * sin(phi);
}
void SDL::getEdgeHits(unsigned int detId,float x, float y, float& xhigh, float& yhigh, float& xlow, float& ylow)
{
    float phi = endcapGeometry.getCentroidPhi(detId);
    xhigh = x + 2.5 * cos(phi);
    yhigh = y + 2.5 * sin(phi);
    xlow = x - 2.5 * cos(phi);
    ylow = y - 2.5 * sin(phi);
}

void SDL::printHit(struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int hitIndex)
{
    std::cout << "Hit(x=" << hitsInGPU.xs[hitIndex] << ", y=" << hitsInGPU.ys[hitIndex] << ", z=" << hitsInGPU.zs[hitIndex] << ", rt=" << hitsInGPU.rts[hitIndex] << ", phi=" << hitsInGPU.phis[hitIndex] <<", module subdet = "<<modulesInGPU.subdets[hitsInGPU.moduleIndices[hitIndex]]<<", module layer = "<< modulesInGPU.layers[hitsInGPU.moduleIndices[hitIndex]]<<", module ring = "<< modulesInGPU.rings[hitsInGPU.moduleIndices[hitIndex]]<<" )"<<std::endl;
}


void SDL::hits::freeMemoryCache()
{
#ifdef Explicit_Hit
    int dev;
    cudaGetDevice(&dev);
    cms::cuda::free_device(dev,nHits);
    cms::cuda::free_device(dev,xs);
    cms::cuda::free_device(dev,ys);
    cms::cuda::free_device(dev,zs);
    cms::cuda::free_device(dev,moduleIndices);
    cms::cuda::free_device(dev,rts);
    cms::cuda::free_device(dev,idxs);
    cms::cuda::free_device(dev,phis);
    cms::cuda::free_device(dev,etas);

    cms::cuda::free_device(dev,highEdgeXs);
    cms::cuda::free_device(dev,highEdgeYs);
    cms::cuda::free_device(dev,lowEdgeXs);
    cms::cuda::free_device(dev,lowEdgeYs);
#else
    cms::cuda::free_managed(nHits);
    cms::cuda::free_managed(xs);
    cms::cuda::free_managed(ys);
    cms::cuda::free_managed(zs);
    cms::cuda::free_managed(moduleIndices);
    cms::cuda::free_managed(rts);
    cms::cuda::free_managed(idxs);
    cms::cuda::free_managed(phis);
    cms::cuda::free_managed(etas);

    cms::cuda::free_managed(highEdgeXs);
    cms::cuda::free_managed(highEdgeYs);
    cms::cuda::free_managed(lowEdgeXs);
    cms::cuda::free_managed(lowEdgeYs);
#endif
}
void SDL::hits::freeMemory(cudaStream_t stream)
{
    cudaFree(nHits);
    cudaFree(xs);
    cudaFree(ys);
    cudaFree(zs);
    cudaFree(moduleIndices);
    cudaFree(rts);
    cudaFree(idxs);
    cudaFree(phis);
    cudaFree(etas);

    cudaFree(highEdgeXs);
    cudaFree(highEdgeYs);
    cudaFree(lowEdgeXs);
    cudaFree(lowEdgeYs);
    cudaStreamSynchronize(stream);
}
