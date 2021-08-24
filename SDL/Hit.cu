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
//    edge2SMap = nullptr;
    highEdgeXs = nullptr;
    highEdgeYs = nullptr;
    lowEdgeXs = nullptr;
    lowEdgeYs = nullptr;
}

SDL::hits::~hits()
{
}
//FIXME:New array!
void SDL::createHitsInUnifiedMemory(struct hits& hitsInGPU,unsigned int nMaxHits,unsigned int nMax2SHits)
{
#ifdef CACHE_ALLOC
    cudaStream_t stream=0;
    hitsInGPU.xs = (float*)cms::cuda::allocate_managed(nMaxHits*sizeof(float),stream);
    hitsInGPU.ys = (float*)cms::cuda::allocate_managed(nMaxHits*sizeof(float),stream);
    hitsInGPU.zs = (float*)cms::cuda::allocate_managed(nMaxHits*sizeof(float),stream);

    hitsInGPU.rts = (float*)cms::cuda::allocate_managed(nMaxHits*sizeof(float),stream);
    hitsInGPU.phis = (float*)cms::cuda::allocate_managed(nMaxHits*sizeof(float),stream);
    hitsInGPU.etas = (float*)cms::cuda::allocate_managed(nMaxHits*sizeof(float),stream);

    hitsInGPU.moduleIndices = (unsigned int*)cms::cuda::allocate_managed(nMaxHits*sizeof(unsigned int),stream);
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
    cudaMallocManaged(&hitsInGPU.moduleIndices, nMaxHits * sizeof(unsigned int));
    //TODO:This dude (idxs) is not used in the GPU at all. It is only used for simhit matching to make efficiency plots
    //We can even skip this one later
    cudaMallocManaged(&hitsInGPU.idxs, nMaxHits * sizeof(unsigned int));

    cudaMallocManaged(&hitsInGPU.rts, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.phis, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.etas, nMaxHits * sizeof(float));

//    cudaMallocManaged(&hitsInGPU.edge2SMap, nMaxHits * sizeof(int)); //hits to edge hits map. Signed int
    //cudaMallocManaged(&hitsInGPU.highEdgeXs, nMax2SHits * sizeof(float)); // due to changes made for the explicit version
    //cudaMallocManaged(&hitsInGPU.highEdgeYs, nMax2SHits * sizeof(float)); // higher memory usage but timing is comparable and results are correct.
    //cudaMallocManaged(&hitsInGPU.lowEdgeXs, nMax2SHits * sizeof(float));
    //cudaMallocManaged(&hitsInGPU.lowEdgeYs, nMax2SHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.highEdgeXs, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.highEdgeYs, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.lowEdgeXs, nMaxHits * sizeof(float));
    cudaMallocManaged(&hitsInGPU.lowEdgeYs, nMaxHits * sizeof(float));

    //counters
    cudaMallocManaged(&hitsInGPU.nHits, sizeof(unsigned int));
#endif
    *hitsInGPU.nHits = 0;
//    cudaMallocManaged(&hitsInGPU.n2SHits, sizeof(unsigned int));
//    *hitsInGPU.n2SHits = 0;
}
void SDL::createHitsInExplicitMemory(struct hits& hitsInGPU, unsigned int nMaxHits)
{
#ifdef CACHE_ALLOC
    cudaStream_t stream=0;
    int dev;
    cudaGetDevice(&dev);
    hitsInGPU.xs = (float*)cms::cuda::allocate_device(dev,nMaxHits*sizeof(float),stream);
    hitsInGPU.ys = (float*)cms::cuda::allocate_device(dev,nMaxHits*sizeof(float),stream);
    hitsInGPU.zs = (float*)cms::cuda::allocate_device(dev,nMaxHits*sizeof(float),stream);

    hitsInGPU.rts = (float*)cms::cuda::allocate_device(dev,nMaxHits*sizeof(float),stream);
    hitsInGPU.phis = (float*)cms::cuda::allocate_device(dev,nMaxHits*sizeof(float),stream);
    hitsInGPU.etas = (float*)cms::cuda::allocate_device(dev,nMaxHits*sizeof(float),stream);

    hitsInGPU.moduleIndices = (unsigned int*)cms::cuda::allocate_device(dev,nMaxHits*sizeof(unsigned int),stream);
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

    cudaMalloc(&hitsInGPU.moduleIndices, nMaxHits * sizeof(unsigned int));
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
    cudaMemset(hitsInGPU.nHits,0,sizeof(unsigned int));
}

__global__ void SDL::addHitToMemoryGPU(struct hits& hitsInCPU, struct modules& modulesInGPU, float x, float y, float z, unsigned int detId, unsigned int idxInNtuple,unsigned int moduleIndex,float phis)
{
    unsigned int idx = *(hitsInCPU.nHits);
//    unsigned int idxEdge2S = *(hitsInCPU.n2SHits);

    hitsInCPU.xs[idx] = x;
    hitsInCPU.ys[idx] = y;
    hitsInCPU.zs[idx] = z;
    hitsInCPU.rts[idx] = sqrt(x*x + y*y);
    hitsInCPU.phis[idx] = phi(x,y,z);
    hitsInCPU.idxs[idx] = idxInNtuple;
 //   unsigned int moduleIndex = (*detIdToIndex)[detId];
    hitsInCPU.moduleIndices[idx] = moduleIndex;
    if(modulesInGPU.subdets[moduleIndex] == Endcap and modulesInGPU.moduleType[moduleIndex] == TwoS)
    {
        float xhigh, yhigh, xlow, ylow;
        //getEdgeHits(detId,x,y,xhigh,yhigh,xlow,ylow);
        getEdgeHitsK(phis,x,y,xhigh,yhigh,xlow,ylow);
        //hitsInCPU.edge2SMap[idx] = idxEdge2S;
        //hitsInCPU.highEdgeXs[idxEdge2S] = xhigh; // due to changes to support the explicit version
        //hitsInCPU.highEdgeYs[idxEdge2S] = yhigh; 
        //hitsInCPU.lowEdgeXs[idxEdge2S] = xlow;
        //hitsInCPU.lowEdgeYs[idxEdge2S] = ylow;
        hitsInCPU.highEdgeXs[idx] = xhigh;
        hitsInCPU.highEdgeYs[idx] = yhigh;
        hitsInCPU.lowEdgeXs[idx] = xlow;
        hitsInCPU.lowEdgeYs[idx] = ylow;

        //(*hitsInCPU.n2SHits)++;
    }
//    else
//    {
//        hitsInCPU.edge2SMap[idx] = -1;
//    }

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
void SDL::addHitToMemory(struct hits& hitsInGPU, struct modules& modulesInGPU, float x, float y, float z, unsigned int detId, unsigned int idxInNtuple)
{
    unsigned int idx = *(hitsInGPU.nHits);
//    unsigned int idxEdge2S = *(hitsInCPU.n2SHits);

    hitsInGPU.xs[idx] = x;
    hitsInGPU.ys[idx] = y;
    hitsInGPU.zs[idx] = z;
    hitsInGPU.rts[idx] = sqrt(x*x + y*y);
    hitsInGPU.phis[idx] = phi(x,y,z);
    hitsInGPU.idxs[idx] = idxInNtuple;
    unsigned int moduleIndex = (*detIdToIndex)[detId];
    hitsInGPU.moduleIndices[idx] = moduleIndex;
    
    unsigned int nModules;
    cudaMemcpy(&nModules,modulesInGPU.nModules,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    
    ModuleType* module_moduleType;
    cudaMallocHost(&module_moduleType, nModules* sizeof(ModuleType));
    cudaMemcpy(module_moduleType,modulesInGPU.moduleType,nModules*sizeof(ModuleType),cudaMemcpyDeviceToHost);
    short* module_subdets;
    cudaMallocHost(&module_subdets, nModules* sizeof(short));
    cudaMemcpy(module_subdets,modulesInGPU.subdets,nModules*sizeof(short),cudaMemcpyDeviceToHost);
    int* module_hitRanges;
    cudaMallocHost(&module_hitRanges, nModules* 2*sizeof(int));
    cudaMemcpy(module_hitRanges,modulesInGPU.hitRanges,nModules*2*sizeof(int),cudaMemcpyDeviceToHost);

    if(module_subdets[moduleIndex] == Endcap and module_moduleType[moduleIndex] == TwoS)
    {
        float xhigh, yhigh, xlow, ylow;
        getEdgeHits(detId,x,y,xhigh,yhigh,xlow,ylow);
        //hitsInCPU.edge2SMap[idx] = idxEdge2S;
        //hitsInCPU.highEdgeXs[idxEdge2S] = xhigh; // due to changes to support the explicit version
        //hitsInCPU.highEdgeYs[idxEdge2S] = yhigh; 
        //hitsInCPU.lowEdgeXs[idxEdge2S] = xlow;
        //hitsInCPU.lowEdgeYs[idxEdge2S] = ylow;
        hitsInGPU.highEdgeXs[idx] = xhigh;
        hitsInGPU.highEdgeYs[idx] = yhigh;
        hitsInGPU.lowEdgeXs[idx] = xlow;
        hitsInGPU.lowEdgeYs[idx] = ylow;

        //(*hitsInCPU.n2SHits)++;
    }
//    else
//    {
//        hitsInCPU.edge2SMap[idx] = -1;
//    }

    //set the hit ranges appropriately in the modules struct

    //start the index rolling if the module is encountered for the first time
    if(module_hitRanges[moduleIndex * 2] == -1)
    {
        module_hitRanges[moduleIndex * 2] = idx;
    }
    //always update the end index
    module_hitRanges[moduleIndex * 2 + 1] = idx;
    cudaMemcpy(modulesInGPU.hitRanges,module_hitRanges,nModules*2*sizeof(int),cudaMemcpyHostToDevice); 
    cudaFreeHost(module_moduleType);
    cudaFreeHost(module_subdets);
    cudaFreeHost(module_hitRanges);
   (*hitsInGPU.nHits)++;
}
__global__ void SDL::addHitToMemoryKernel(struct hits& hitsInGPU, struct modules& modulesInGPU,const float* x,const  float* y,const  float* z, const unsigned int* moduleIndex,const float* phis, const int loopsize)
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
    //  if(modulesInGPU.subdets[moduleIndex[ihit]] == Endcap && modulesInGPU.moduleType[moduleIndex[ihit]] == TwoS)
    //  {
    //      float xhigh, yhigh, xlow, ylow;
    //      getEdgeHitsK(phis[ihit],x[ihit],y[ihit],xhigh,yhigh,xlow,ylow);
    ////      hitsInGPU.edge2SMap[idx] = idxEdge2S;
    //      //hitsInGPU.highEdgeXs[idxEdge2S] = xhigh;
    //      //hitsInGPU.highEdgeYs[idxEdge2S] = yhigh;
    //      //hitsInGPU.lowEdgeXs[idxEdge2S] = xlow;
    //      //hitsInGPU.lowEdgeYs[idxEdge2S] = ylow;
    //      hitsInGPU.highEdgeXs[idx] = xhigh;
    //      hitsInGPU.highEdgeYs[idx] = yhigh;
    //      hitsInGPU.lowEdgeXs[idx] = xlow;
    //      hitsInGPU.lowEdgeYs[idx] = ylow;
  
    //   //   (*hitsInGPU.n2SHits)++;
    //  }
  
    //  //set the hit ranges appropriately in the modules struct
  
    //  //start the index rolling if the module is encountered for the first time
    //  if(modulesInGPU.hitRanges[moduleIndex[ihit] * 2] == -1)
    //  {
    //      modulesInGPU.hitRanges[moduleIndex[ihit] * 2] = idx;
    //  }
    //  //always update the end index
    //  modulesInGPU.hitRanges[moduleIndex[ihit] * 2 + 1] = idx;
  }
}
//__global__ void SDL::checkHits(struct hits& hitsInGPU, const int loopsize){
//  //for (unsigned int ihit = blockIdx.x*blockDim.x + threadIdx.x; ihit <loopsize; ihit += blockDim.x*gridDim.x)
//  for (int ihit = 0; ihit <loopsize; ihit ++ )
//  {
//    printf("checkHits: %d %f %f %f %f %f %u %u %f %f %f %f\n",ihit,hitsInGPU.xs[ihit],hitsInGPU.ys[ihit],hitsInGPU.zs[ihit],hitsInGPU.rts[ihit],hitsInGPU.phis[ihit],hitsInGPU.moduleIndices[ihit],hitsInGPU.idxs[ihit],hitsInGPU.highEdgeXs[ihit],hitsInGPU.highEdgeYs[ihit],hitsInGPU.lowEdgeXs[ihit],hitsInGPU.lowEdgeYs[ihit]);
//  }
//}
float SDL::ATan2(float y, float x)
{
    if (x != 0) return  atan2(y, x);
    if (y == 0) return  0;
    if (y >  0) return  M_PI / 2;
    else        return -M_PI / 2;
}

//TODO:Check if cuda atan2f will work here
float SDL::phi(float x, float y, float z)
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
//    cudaFree(n2SHits);
    cudaFree(xs);
    cudaFree(ys);
    cudaFree(zs);
    cudaFree(moduleIndices);
    cudaFree(rts);
    cudaFree(idxs);
    cudaFree(phis);
    cudaFree(etas);

//    cudaFree(edge2SMap);
    cudaFree(highEdgeXs);
    cudaFree(highEdgeYs);
    cudaFree(lowEdgeXs);
    cudaFree(lowEdgeYs);
}
