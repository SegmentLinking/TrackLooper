# include "Hit.cuh"
#ifdef __CUDACC__
#define CUDA_CONST_VAR __device__
#endif

SDL::hits::hits()
{
    nHits = nullptr;
    n2SHits = nullptr;
    xs = nullptr;
    ys = nullptr;
    zs = nullptr;
    moduleIndices = nullptr;
    rts = nullptr;
    phis = nullptr;
    edge2SMap = nullptr;
    highEdgeXs = nullptr;
    highEdgeYs = nullptr;
    lowEdgeXs = nullptr;
    lowEdgeYs = nullptr;
}

void SDL::createHitsInUnifiedMemory(struct hits& hitsInGPU,unsigned int nMaxHits,unsigned int nMax2SHits)
{
    //nMaxHits and nMax2SHits are the maximum possible numbers
    cudaMallocManaged(&hitsInGPU.xs, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.ys, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.zs, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.moduleIndices, nMaxHits * sizeof(unsigned int));

    cudaMallocManaged(&hitsInGPU.rts, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.phis, nMaxHits * sizeof(float));

    cudaMallocManaged(&hitsInGPU.edge2SMap, nMaxHits * sizeof(int)); //hits to edge hits map. Signed int
    cudaMallocManaged(&hitsInGPU.highEdgeXs, nMax2SHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.highEdgeYs, nMax2SHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.lowEdgeXs, nMax2SHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.lowEdgeYs, nMax2SHits * sizeof(float));

    //counters
    cudaMallocManaged(&hitsInGPU.nHits, sizeof(unsigned int));
    *hitsInGPU.nHits = 0;
    cudaMallocManaged(&hitsInGPU.n2SHits, sizeof(unsigned int));
    *hitsInGPU.n2SHits = 0;
}
void SDL::createHitsInExplicitMemory(struct hits& hitsInGPU, unsigned int nMaxHits,unsigned int nMax2SHits)
{
    cudaMalloc(&hitsInGPU.xs, nMaxHits * sizeof(float));
    cudaMalloc(&hitsInGPU.ys, nMaxHits * sizeof(float));
    cudaMalloc(&hitsInGPU.zs, nMaxHits * sizeof(float));
    cudaMalloc(&hitsInGPU.moduleIndices, nMaxHits * sizeof(unsigned int));

    cudaMalloc(&hitsInGPU.rts, nMaxHits * sizeof(float));
    cudaMalloc(&hitsInGPU.phis, nMaxHits * sizeof(float));

    cudaMalloc(&hitsInGPU.edge2SMap, nMaxHits * sizeof(int)); //hits to edge hits map. Signed int
    cudaMalloc(&hitsInGPU.highEdgeXs, nMax2SHits * sizeof(float));
    cudaMalloc(&hitsInGPU.highEdgeYs, nMax2SHits * sizeof(float));
    cudaMalloc(&hitsInGPU.lowEdgeXs, nMax2SHits * sizeof(float));
    cudaMalloc(&hitsInGPU.lowEdgeYs, nMax2SHits * sizeof(float));

    //counters
    cudaMallocManaged(&hitsInGPU.nHits, sizeof(unsigned int));
    *hitsInGPU.nHits = 0;
    cudaMallocManaged(&hitsInGPU.n2SHits, sizeof(unsigned int));
    *hitsInGPU.n2SHits = 0;
}

void SDL::addHitToMemory(struct hits& hitsInCPU, struct modules& modulesInGPU, float x, float y, float z, unsigned int detId)
{
    unsigned int idx = *(hitsInCPU.nHits);
    unsigned int idxEdge2S = *(hitsInCPU.n2SHits);

    hitsInCPU.xs[idx] = x;
    hitsInCPU.ys[idx] = y;
    hitsInCPU.zs[idx] = z;
    hitsInCPU.rts[idx] = sqrt(x*x + y*y);
    hitsInCPU.phis[idx] = phi(x,y,z);
    unsigned int moduleIndex = (*detIdToIndex)[detId];
    hitsInCPU.moduleIndices[idx] = moduleIndex;
    if(modulesInGPU.subdets[moduleIndex] == Endcap and modulesInGPU.moduleType[moduleIndex] == TwoS)
    {
        float xhigh, yhigh, xlow, ylow;
        getEdgeHits(detId,x,y,xhigh,yhigh,xlow,ylow);
        hitsInCPU.edge2SMap[idx] = idxEdge2S;
        hitsInCPU.highEdgeXs[idxEdge2S] = xhigh;
        hitsInCPU.highEdgeYs[idxEdge2S] = yhigh;
        hitsInCPU.lowEdgeXs[idxEdge2S] = xlow;
        hitsInCPU.lowEdgeYs[idxEdge2S] = ylow;

        (*hitsInCPU.n2SHits)++;
    }
    else
    {
        hitsInCPU.edge2SMap[idx] = -1;
    }

    //set the hit ranges appropriately in the modules struct

    //start the index rolling if the module is encountered for the first time
    if(modulesInGPU.hitRanges[moduleIndex * 2] == -1)
    {
        modulesInGPU.hitRanges[moduleIndex * 2] = idx;
    }
    //always update the end index
    modulesInGPU.hitRanges[moduleIndex * 2 + 1] = idx;
    (*hitsInCPU.nHits)++;
}
__global__ void SDL::addHitToMemoryKernel(struct hits& hitsInGPU, struct modules& modulesInGPU,const float* x,const  float* y,const  float* z, const unsigned int* detId, const unsigned int* moduleIndex,const float* phis, const int loopsize)
{
  for (unsigned int ihit = blockIdx.x*blockDim.x + threadIdx.x; ihit <loopsize; ihit += blockDim.x*gridDim.x)
  {
      unsigned int idx = ihit;//*(hitsInGPU.nHits);
      unsigned int idxEdge2S = *(hitsInGPU.n2SHits);
  
      hitsInGPU.xs[idx] = x[ihit];
      hitsInGPU.ys[idx] = y[ihit];
      hitsInGPU.zs[idx] = z[ihit];
      hitsInGPU.rts[idx] = sqrt(x[ihit]*x[ihit] + y[ihit]*y[ihit]);
      hitsInGPU.phis[idx] = phi(x[ihit],y[ihit],z[ihit]);
      hitsInGPU.moduleIndices[idx] = moduleIndex[ihit];
      if(modulesInGPU.subdets[moduleIndex[ihit]] == Endcap and modulesInGPU.moduleType[moduleIndex[ihit]] == TwoS)
      {
          float xhigh, yhigh, xlow, ylow;
          getEdgeHitsK(phis[ihit],x[ihit],y[ihit],xhigh,yhigh,xlow,ylow);
          hitsInGPU.edge2SMap[idx] = idxEdge2S;
          hitsInGPU.highEdgeXs[idxEdge2S] = xhigh;
          hitsInGPU.highEdgeYs[idxEdge2S] = yhigh;
          hitsInGPU.lowEdgeXs[idxEdge2S] = xlow;
          hitsInGPU.lowEdgeYs[idxEdge2S] = ylow;
  
          (*hitsInGPU.n2SHits)++;
      }
      else
      {
          hitsInGPU.edge2SMap[idx] = -1;
      }
  
      //set the hit ranges appropriately in the modules struct
  
      //start the index rolling if the module is encountered for the first time
      if(modulesInGPU.hitRanges[moduleIndex[ihit] * 2] == -1)
      {
          modulesInGPU.hitRanges[moduleIndex[ihit] * 2] = idx;
      }
      //always update the end index
      modulesInGPU.hitRanges[moduleIndex[ihit] * 2 + 1] = idx;
      (*hitsInGPU.nHits)++;
      //(*hitsInGPU.nHits) = atomicAdd(hitsInGPU.nHits,1);
  }
}

inline float SDL::ATan2(float y, float x)
{
    if (x != 0) return  atan2(y, x);
    if (y == 0) return  0;
    if (y >  0) return  M_PI / 2;
    else        return -M_PI / 2;
}

//TODO:Check if cuda atan2f will work here
inline float SDL::phi(float x, float y, float z)
{
    return phi_mpi_pi(M_PI + ATan2(-y, -x)); 
}


float SDL::phi_mpi_pi(float x)
{
    if (isnan(x))
    {
       printf("phi_mpi_pi() function called with NaN\n");
        return x;
    }

    while (x >= M_PI)
        x -= 2. * M_PI;

    while (x < -M_PI)
        x += 2. * M_PI;

    return x;
}

float SDL::deltaPhi(float x1, float y1, float z1, float x2, float y2, float z2)
{
    float phi1 = phi(x1,y1,z1);
    float phi2 = phi(x2,y2,z2);
    return phi_mpi_pi((phi2 - phi1));
}

float SDL::deltaPhiChange(float x1, float y1, float z1, float x2, float y2, float z2)
{
    return deltaPhi(x1,y1,z1,x2-x1, y2-y1, z2-z1);
}

__device__ void SDL::getEdgeHitsK(float phi,float x, float y, float& xhigh, float& yhigh, float& xlow, float& ylow)
{
//    float phi = endcapGeometry.getCentroidPhi(detId);
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


void SDL::hits::freeMemory()
{
    cudaFree(nHits);
    cudaFree(n2SHits);
    cudaFree(xs);
    cudaFree(ys);
    cudaFree(zs);
    cudaFree(moduleIndices);
    cudaFree(rts);
    cudaFree(phis);

    cudaFree(edge2SMap);
    cudaFree(highEdgeXs);
    cudaFree(highEdgeYs);
    cudaFree(lowEdgeXs);
    cudaFree(lowEdgeYs);
}
