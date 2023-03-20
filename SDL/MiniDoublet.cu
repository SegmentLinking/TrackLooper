# include "MiniDoublet.cuh"

void SDL::miniDoublets::resetMemory(unsigned int nMemoryLocationsx, unsigned int nLowerModules,cudaStream_t stream)

{
    cudaMemsetAsync(anchorHitIndices,0, nMemoryLocationsx * 3 * sizeof(unsigned int),stream);
    cudaMemsetAsync(dphichanges,0, nMemoryLocationsx * 9 * sizeof(float),stream);
    cudaMemsetAsync(nMDs,0, (nLowerModules + 1) * sizeof(int),stream);
    cudaMemsetAsync(totOccupancyMDs,0, (nLowerModules + 1) * sizeof(unsigned int),stream);
}

    //#ifdef CUT_VALUE_DEBUG
//ALPAKA_FN_ACC void addMDToMemory(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int lowerHitIdx, unsigned int upperHitIdx, uint16_t& lowerModuleIdx, float dz, float drt, float dPhi, float dPhiChange, float shiftedX, float shiftedY, float shiftedZ, float noShiftedDz, float noShiftedDphi, float noShiftedDPhiChange, float dzCut, float drtCut, float miniCut, unsigned int idx)
//#else
ALPAKA_FN_ACC void SDL::addMDToMemory(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, struct modules& modulesInGPU, unsigned int lowerHitIdx, unsigned int upperHitIdx, uint16_t& lowerModuleIdx, float dz, float dPhi, float dPhiChange, float shiftedX, float shiftedY, float shiftedZ, float noShiftedDz, float noShiftedDphi, float noShiftedDPhiChange, unsigned int idx)
//#endif
{
    //the index into which this MD needs to be written will be computed in the kernel
    //nMDs variable will be incremented in the kernel, no need to worry about that here
    
    mdsInGPU.moduleIndices[idx] = lowerModuleIdx;
    unsigned int anchorHitIndex, outerHitIndex;
    if(modulesInGPU.moduleType[lowerModuleIdx] == PS and modulesInGPU.moduleLayerType[lowerModuleIdx] == Strip)
    {
        mdsInGPU.anchorHitIndices[idx] = upperHitIdx;
        mdsInGPU.outerHitIndices[idx] = lowerHitIdx;

        anchorHitIndex = upperHitIdx;
        outerHitIndex = lowerHitIdx;
    }
    else
    {
        mdsInGPU.anchorHitIndices[idx] = lowerHitIdx;
        mdsInGPU.outerHitIndices[idx] = upperHitIdx;

        anchorHitIndex = lowerHitIdx;
        outerHitIndex = upperHitIdx;
    }

    mdsInGPU.dphichanges[idx] = dPhiChange;

    mdsInGPU.dphis[idx] = dPhi;
    mdsInGPU.dzs[idx] = dz;
    mdsInGPU.shiftedXs[idx] = shiftedX;
    mdsInGPU.shiftedYs[idx] = shiftedY;
    mdsInGPU.shiftedZs[idx] = shiftedZ;

    mdsInGPU.noShiftedDzs[idx] = noShiftedDz;
    mdsInGPU.noShiftedDphis[idx] = noShiftedDphi;
    mdsInGPU.noShiftedDphiChanges[idx] = noShiftedDPhiChange;
//#ifdef CUT_VALUE_DEBUG
//    mdsInGPU.dzCuts[idx] = dzCut;
//    mdsInGPU.drtCuts[idx] = drtCut;
//    mdsInGPU.miniCuts[idx] = miniCut;
//#endif

    mdsInGPU.anchorX[idx] = hitsInGPU.xs[anchorHitIndex];
    mdsInGPU.anchorY[idx] = hitsInGPU.ys[anchorHitIndex];
    mdsInGPU.anchorZ[idx] = hitsInGPU.zs[anchorHitIndex];
    mdsInGPU.anchorRt[idx] = hitsInGPU.rts[anchorHitIndex];
    mdsInGPU.anchorPhi[idx] = hitsInGPU.phis[anchorHitIndex];
    mdsInGPU.anchorEta[idx] = hitsInGPU.etas[anchorHitIndex];
    mdsInGPU.anchorHighEdgeX[idx] = hitsInGPU.highEdgeXs[anchorHitIndex];
    mdsInGPU.anchorHighEdgeY[idx] = hitsInGPU.highEdgeYs[anchorHitIndex];
    mdsInGPU.anchorLowEdgeX[idx] = hitsInGPU.lowEdgeXs[anchorHitIndex];
    mdsInGPU.anchorLowEdgeY[idx] = hitsInGPU.lowEdgeYs[anchorHitIndex];

    mdsInGPU.outerX[idx] = hitsInGPU.xs[outerHitIndex];
    mdsInGPU.outerY[idx] = hitsInGPU.ys[outerHitIndex];
    mdsInGPU.outerZ[idx] = hitsInGPU.zs[outerHitIndex];
    mdsInGPU.outerRt[idx] = hitsInGPU.rts[outerHitIndex];
    mdsInGPU.outerPhi[idx] = hitsInGPU.phis[outerHitIndex];
    mdsInGPU.outerEta[idx] = hitsInGPU.etas[outerHitIndex];
    mdsInGPU.outerHighEdgeX[idx] = hitsInGPU.highEdgeXs[outerHitIndex];
    mdsInGPU.outerHighEdgeY[idx] = hitsInGPU.highEdgeYs[outerHitIndex];
    mdsInGPU.outerLowEdgeX[idx] = hitsInGPU.lowEdgeXs[outerHitIndex];
    mdsInGPU.outerLowEdgeY[idx] = hitsInGPU.lowEdgeYs[outerHitIndex];
}

__global__ void SDL::createMDArrayRangesGPU(struct modules& modulesInGPU, struct objectRanges& rangesInGPU)//, unsigned int* nTotalMDsx)
{
    short module_subdets;
    short module_layers;
    short module_rings;
    float module_eta;

    __shared__ unsigned int nTotalMDs; //start!   
    nTotalMDs = 0; //start!   
    __syncthreads();
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int np = gridDim.x * blockDim.x;
    for(uint16_t i = gid; i < *modulesInGPU.nLowerModules; i+= np)
    {
        module_subdets = modulesInGPU.subdets[i];
        module_layers = modulesInGPU.layers[i];
        module_rings = modulesInGPU.rings[i];
        module_eta = abs(modulesInGPU.eta[i]);
        unsigned int occupancy;
        unsigned int category_number, eta_number;
        if (module_layers<=3 && module_subdets==5) category_number = 0;
        else if (module_layers>=4 && module_subdets==5) category_number = 1;
        else if (module_layers<=2 && module_subdets==4 && module_rings>=11) category_number = 2;
        else if (module_layers>=3 && module_subdets==4 && module_rings>=8) category_number = 2;
        else if (module_layers<=2 && module_subdets==4 && module_rings<=10) category_number = 3;
        else if (module_layers>=3 && module_subdets==4 && module_rings<=7) category_number = 3;

        if (module_eta<0.75) eta_number=0;
        else if (module_eta>0.75 && module_eta<1.5) eta_number=1;
        else if (module_eta>1.5  && module_eta<2.25) eta_number=2;
        else if (module_eta>2.25 && module_eta<3) eta_number=3;

        if (category_number == 0 && eta_number == 0) occupancy = 49;
        else if (category_number == 0 && eta_number == 1) occupancy = 42;
        else if (category_number == 0 && eta_number == 2) occupancy = 37;
        else if (category_number == 0 && eta_number == 3) occupancy = 41;
        else if (category_number == 1) occupancy = 100;
        else if (category_number == 2 && eta_number == 1) occupancy = 16;
        else if (category_number == 2 && eta_number == 2) occupancy = 19;
        else if (category_number == 3 && eta_number == 1) occupancy = 14;
        else if (category_number == 3 && eta_number == 2) occupancy = 20;
        else if (category_number == 3 && eta_number == 3) occupancy = 25;

        unsigned int nTotMDs= atomicAdd(&nTotalMDs,occupancy);
        rangesInGPU.miniDoubletModuleIndices[i] = nTotMDs; 
        rangesInGPU.miniDoubletModuleOccupancy[i] = occupancy;
    }
    __syncthreads();
    if(threadIdx.x==0){
      rangesInGPU.miniDoubletModuleIndices[*modulesInGPU.nLowerModules] = nTotalMDs;
      //*nTotalMDsx=nTotalMDs;
      *rangesInGPU.device_nTotalMDs=nTotalMDs;
    }

}

//FIXME:Add memory locations for the pixel MDs here!
void SDL::createMDsInExplicitMemory(struct miniDoublets& mdsInGPU, unsigned int nMemoryLocations, uint16_t nLowerModules, unsigned int maxPixelMDs,cudaStream_t stream)
{
#ifdef CACHE_ALLOC
//    cudaStream_t stream=0;
    int dev;
    cudaGetDevice(&dev);
    mdsInGPU.anchorHitIndices = (unsigned int*)cms::cuda::allocate_device(dev,nMemoryLocations * 2 * sizeof(unsigned int), stream);
    mdsInGPU.moduleIndices = (uint16_t*)cms::cuda::allocate_device(dev, nMemoryLocations * sizeof(uint16_t), stream);
    mdsInGPU.dphichanges = (float*)cms::cuda::allocate_device(dev,nMemoryLocations*9*sizeof(float),stream);
    mdsInGPU.nMDs = (int*)cms::cuda::allocate_device(dev, (nLowerModules + 1) *sizeof(int),stream);
    mdsInGPU.totOccupancyMDs = (int*)cms::cuda::allocate_device(dev, (nLowerModules + 1) *sizeof(int),stream);
    mdsInGPU.anchorX = (float*)cms::cuda::allocate_device(dev, nMemoryLocations * 6 * sizeof(float), stream);
    mdsInGPU.anchorHighEdgeX = (float*)cms::cuda::allocate_device(dev, nMemoryLocations * 4 * sizeof(float), stream);
    mdsInGPU.outerX = (float*)cms::cuda::allocate_device(dev, nMemoryLocations * 6 * sizeof(float), stream);
    mdsInGPU.outerHighEdgeX = (float*)cms::cuda::allocate_device(dev, nMemoryLocations * 4 * sizeof(float), stream);

    mdsInGPU.nMemoryLocations = (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int), stream);

#else
    cudaMalloc(&mdsInGPU.anchorHitIndices, nMemoryLocations * 2 * sizeof(unsigned int));
    cudaMalloc(&mdsInGPU.moduleIndices, nMemoryLocations * sizeof(uint16_t));
    cudaMalloc(&mdsInGPU.dphichanges, nMemoryLocations *9* sizeof(float));
    cudaMalloc(&mdsInGPU.nMDs, (nLowerModules + 1) * sizeof(int)); 
    cudaMalloc(&mdsInGPU.totOccupancyMDs, (nLowerModules + 1) * sizeof(int)); 
    cudaMalloc(&mdsInGPU.anchorX, nMemoryLocations * 6 * sizeof(float));
    cudaMalloc(&mdsInGPU.anchorHighEdgeX, nMemoryLocations * 4 * sizeof(float));
    cudaMalloc(&mdsInGPU.outerX, nMemoryLocations * 6 * sizeof(float));
    cudaMalloc(&mdsInGPU.outerHighEdgeX, nMemoryLocations * 4 * sizeof(float));

    cudaMalloc(&mdsInGPU.nMemoryLocations, sizeof(unsigned int));

#endif
    cudaMemsetAsync(mdsInGPU.nMDs,0, (nLowerModules + 1) *sizeof(int),stream);
    cudaMemsetAsync(mdsInGPU.totOccupancyMDs,0, (nLowerModules + 1) *sizeof(int),stream);
    cudaStreamSynchronize(stream);

    mdsInGPU.outerHitIndices = mdsInGPU.anchorHitIndices + nMemoryLocations;
    mdsInGPU.dzs  = mdsInGPU.dphichanges + nMemoryLocations;
    mdsInGPU.dphis  = mdsInGPU.dphichanges + 2*nMemoryLocations;
    mdsInGPU.shiftedXs  = mdsInGPU.dphichanges + 3*nMemoryLocations;
    mdsInGPU.shiftedYs  = mdsInGPU.dphichanges + 4*nMemoryLocations;
    mdsInGPU.shiftedZs  = mdsInGPU.dphichanges + 5*nMemoryLocations;
    mdsInGPU.noShiftedDzs  = mdsInGPU.dphichanges + 6*nMemoryLocations;
    mdsInGPU.noShiftedDphis  = mdsInGPU.dphichanges + 7*nMemoryLocations;
    mdsInGPU.noShiftedDphiChanges  = mdsInGPU.dphichanges + 8*nMemoryLocations;

    mdsInGPU.anchorY = mdsInGPU.anchorX + nMemoryLocations;
    mdsInGPU.anchorZ = mdsInGPU.anchorX + 2 * nMemoryLocations;
    mdsInGPU.anchorRt = mdsInGPU.anchorX + 3 * nMemoryLocations;
    mdsInGPU.anchorPhi = mdsInGPU.anchorX + 4 * nMemoryLocations;
    mdsInGPU.anchorEta = mdsInGPU.anchorX + 5 * nMemoryLocations;

    mdsInGPU.anchorHighEdgeY = mdsInGPU.anchorHighEdgeX + nMemoryLocations;
    mdsInGPU.anchorLowEdgeX = mdsInGPU.anchorHighEdgeX + 2 * nMemoryLocations;
    mdsInGPU.anchorLowEdgeY = mdsInGPU.anchorHighEdgeX + 3 * nMemoryLocations;

    mdsInGPU.outerY = mdsInGPU.outerX + nMemoryLocations;
    mdsInGPU.outerZ = mdsInGPU.outerX + 2 * nMemoryLocations;
    mdsInGPU.outerRt = mdsInGPU.outerX + 3 * nMemoryLocations;
    mdsInGPU.outerPhi = mdsInGPU.outerX + 4 * nMemoryLocations;
    mdsInGPU.outerEta = mdsInGPU.outerX + 5 * nMemoryLocations;

    mdsInGPU.outerHighEdgeY = mdsInGPU.outerHighEdgeX + nMemoryLocations;
    mdsInGPU.outerLowEdgeX = mdsInGPU.outerHighEdgeX + 2 * nMemoryLocations;
    mdsInGPU.outerLowEdgeY = mdsInGPU.outerHighEdgeX + 3 * nMemoryLocations;
}

ALPAKA_FN_ACC void SDL::shiftStripHits(struct modules& modulesInGPU, /*struct hits& hitsInGPU,*/ uint16_t& lowerModuleIndex, uint16_t& upperModuleIndex, unsigned int lowerHitIndex, unsigned int upperHitIndex, float* shiftedCoords, float xLower, float yLower, float zLower, float rtLower,float xUpper,float yUpper,float zUpper,float rtUpper)
{

    // This is the strip shift scheme that is explained in http://uaf-10.t2.ucsd.edu/~phchang/talks/PhilipChang20190607_SDL_Update.pdf (see backup slides)
    // The main feature of this shifting is that the strip hits are shifted to be "aligned" in the line of sight from interaction point to the the pixel hit.
    // (since pixel hit is well defined in 3-d)
    // The strip hit is shifted along the strip detector to be placed in a guessed position where we think they would have actually crossed
    // The size of the radial direction shift due to module separation gap is computed in "radial" size, while the shift is done along the actual strip orientation
    // This means that there may be very very subtle edge effects coming from whether the strip hit is center of the module or the at the edge of the module
    // But this should be relatively minor effect

    // dependent variables for this if statement
    // lowerModule
    // lowerHit
    // upperHit
    // SDL::endcapGeometry
    // SDL::tiltedGeometry

    // Some variables relevant to the function
    float xp; // pixel x (pixel hit x)
    float yp; // pixel y (pixel hit y)
    float zp; // pixel y (pixel hit y)
    float rtp; // pixel y (pixel hit y)
    float xa; // "anchor" x (the anchor position on the strip module plane from pixel hit)
    float ya; // "anchor" y (the anchor position on the strip module plane from pixel hit)
    float xo; // old x (before the strip hit is moved up or down)
    float yo; // old y (before the strip hit is moved up or down)
    float xn; // new x (after the strip hit is moved up or down)
    float yn; // new y (after the strip hit is moved up or down)
    float abszn; // new z in absolute value
    float zn; // new z with the sign (+/-) accounted
    float angleA; // in r-z plane the theta of the pixel hit in polar coordinate is the angleA
    float angleB; // this is the angle of tilted module in r-z plane ("drdz"), for endcap this is 90 degrees
    bool isEndcap; // If endcap, drdz = infinity
    //unsigned int pixelHitIndex; // Pointer to the pixel hit
    //unsigned int stripHitIndex; // Pointer to the strip hit
    float moduleSeparation;
    float drprime; // The radial shift size in x-y plane projection
    float drprime_x; // x-component of drprime
    float drprime_y; // y-component of drprime
    float& slope = modulesInGPU.slopes[lowerModuleIndex]; // The slope of the possible strip hits for a given module in x-y plane
    float absArctanSlope;
    float angleM; // the angle M is the angle of rotation of the module in x-y plane if the possible strip hits are along the x-axis, then angleM = 0, and if the possible strip hits are along y-axis angleM = 90 degrees
    float absdzprime; // The distance between the two points after shifting
    float& drdz_ = modulesInGPU.drdzs[lowerModuleIndex];
    // Assign hit pointers based on their hit type
    if (modulesInGPU.moduleType[lowerModuleIndex] == PS)
    {
// TODO: This is somewhat of an mystery.... somewhat confused why this is the case
        if (modulesInGPU.subdets[lowerModuleIndex] == Barrel ? modulesInGPU.moduleLayerType[lowerModuleIndex] != Pixel : modulesInGPU.moduleLayerType[lowerModuleIndex] == Pixel)
        {
            //old to delete
       //     pixelHitIndex = lowerHitIndex;
       //     stripHitIndex = upperHitIndex;
    
            //new to replace
            xo =xUpper;
            yo =yUpper;
            xp =xLower;
            yp =yLower;
            zp =zLower;
            rtp =rtLower;
            xp =xLower;
            yp =yLower;
            zp =zLower;
            rtp =rtLower;
        }
        else
        {
     //       pixelHitIndex = upperHitIndex;
     //       stripHitIndex = lowerHitIndex;
            //new to replace
            xo = xLower;
            yo = yLower;
            xp = xUpper;
            yp = yUpper;
            zp = zUpper;
            rtp=rtUpper;
            xp = xUpper;
            yp = yUpper;
            zp = zUpper;
            rtp=rtUpper;
        }
    }
    else // if (lowerModule.moduleType() == SDL::Module::TwoS) // If it is a TwoS module (if this is called likely an endcap module) then anchor the inner hit and shift the outer hit
    {
        //pixelHitIndex = lowerHitIndex; // Even though in this case the "pixelHitPtr" is really just a strip hit, we pretend it is the anchoring pixel hit
        //stripHitIndex = upperHitIndex;
            xo =xUpper;
            yo =yUpper;
            xp =xLower;
            yp =yLower;
            zp =zLower;
            rtp =rtLower;
            xp =xLower;
            yp =yLower;
            zp =zLower;
            rtp =rtLower;
    }

    // If it is endcap some of the math gets simplified (and also computers don't like infinities)
    isEndcap = modulesInGPU.subdets[lowerModuleIndex]== Endcap;

    // NOTE: TODO: Keep in mind that the sin(atan) function can be simplifed to something like x / sqrt(1 + x^2) and similar for cos
    // I am not sure how slow sin, atan, cos, functions are in c++. If x / sqrt(1 + x^2) are faster change this later to reduce arithmetic computation time

    // The pixel hit is used to compute the angleA which is the theta in polar coordinate
    // angleA = atanf(pixelHitPtr->rt() / pixelHitPtr->z() + (pixelHitPtr->z() < 0 ? M_PI : 0)); // Shift by pi if the z is negative so that the value of the angleA stays between 0 to pi and not -pi/2 to pi/2

    angleA = fabsf(atanf(rtp / zp));
    angleB = ((isEndcap) ? float(M_PI) / 2.f : atanf(drdz_)); // The tilt module on the postive z-axis has negative drdz slope in r-z plane and vice versa


    moduleSeparation = moduleGapSize(modulesInGPU, lowerModuleIndex);

    // Sign flips if the pixel is later layer
    if (modulesInGPU.moduleType[lowerModuleIndex] == PS and modulesInGPU.moduleLayerType[lowerModuleIndex] != Pixel)
    {
        moduleSeparation *= -1;
    }

    drprime = (moduleSeparation / sinf(angleA + angleB)) * sinf(angleA);
    
    // Compute arctan of the slope and take care of the slope = infinity case
    absArctanSlope = ((slope != SDL::SDL_INF) ? fabs(atanf(slope)) : float(M_PI) / 2.f);


    // Depending on which quadrant the pixel hit lies, we define the angleM by shifting them slightly differently
    if (xp > 0 and yp > 0)
    {
        angleM = absArctanSlope;
    }
    else if (xp > 0 and yp < 0)
    {
        angleM = float(M_PI) - absArctanSlope;
    }
    else if (xp < 0 and yp < 0)
    {
        angleM = float(M_PI) + absArctanSlope;
    }
    else // if (xp < 0 and yp > 0)
    {
        angleM = 2.f * float(M_PI) - absArctanSlope;
    }

    // Then since the angleM sign is taken care of properly
    drprime_x = drprime * sinf(angleM);
    drprime_y = drprime * cosf(angleM);

    // The new anchor position is
    xa = xp + drprime_x;
    ya = yp + drprime_y;

    // The original strip hit position
    //xo = hitsInGPU.xs[stripHitIndex];
    //yo = hitsInGPU.ys[stripHitIndex];

    // Compute the new strip hit position (if the slope vaule is in special condition take care of the exceptions)
    if (slope == SDL::SDL_INF) // Designated for tilted module when the slope is exactly infinity (module lying along y-axis)
    {
        xn = xa; // New x point is simply where the anchor is
        yn = yo; // No shift in y
    }
    else if (slope == 0)
    {
        xn = xo; // New x point is simply where the anchor is
        yn = ya; // No shift in y
    }
    else
    {
        xn = (slope * xa + (1.f / slope) * xo - ya + yo) / (slope + (1.f / slope)); // new xn
        yn = (xn - xa) * slope + ya; // new yn
    }

    // Computing new Z position
    absdzprime = fabsf(moduleSeparation / sinf(angleA + angleB) * cosf(angleA)); // module separation sign is for shifting in radial direction for z-axis direction take care of the sign later

    // Depending on which one as closer to the interactin point compute the new z wrt to the pixel properly
    if (modulesInGPU.moduleLayerType[lowerModuleIndex] == Pixel)
    {
        abszn = fabsf(zp) + absdzprime;
    }
    else
    {
        abszn = fabsf(zp) - absdzprime;
    }

    zn = abszn * ((zp > 0) ? 1 : -1); // Apply the sign of the zn


    shiftedCoords[0] = xn;
    shiftedCoords[1] = yn;
    shiftedCoords[2] = zn;
}

SDL::miniDoublets::miniDoublets()
{
    anchorHitIndices = nullptr;
    outerHitIndices = nullptr;
    moduleIndices = nullptr;
    nMDs = nullptr;
    totOccupancyMDs = nullptr;
    dphichanges = nullptr;

    dzs = nullptr;
    dphis = nullptr;

    shiftedXs = nullptr;
    shiftedYs = nullptr;
    shiftedZs = nullptr;
    noShiftedDzs = nullptr;
    noShiftedDphis = nullptr;
    noShiftedDphiChanges = nullptr;
    
    anchorX = nullptr;
    anchorY = nullptr;
    anchorZ = nullptr;
    anchorRt = nullptr;
    anchorPhi = nullptr;
    anchorEta = nullptr;
    anchorHighEdgeX = nullptr;
    anchorHighEdgeY = nullptr;
    anchorLowEdgeX = nullptr;
    anchorLowEdgeY = nullptr;
    outerX = nullptr;
    outerY = nullptr;
    outerZ = nullptr;
    outerRt = nullptr;
    outerPhi = nullptr;
    outerEta = nullptr;
    outerHighEdgeX = nullptr;
    outerHighEdgeY = nullptr;
    outerLowEdgeX = nullptr;
    outerLowEdgeY = nullptr;
}

SDL::miniDoublets::~miniDoublets()
{
}

void SDL::miniDoublets::freeMemoryCache()
{
    int dev;
    cudaGetDevice(&dev);
    cms::cuda::free_device(dev,anchorHitIndices);
    cms::cuda::free_device(dev, moduleIndices);
    cms::cuda::free_device(dev,dphichanges);
    cms::cuda::free_device(dev,nMDs);
    cms::cuda::free_device(dev,totOccupancyMDs);
    cms::cuda::free_device(dev, anchorX);
    cms::cuda::free_device(dev, anchorHighEdgeX);
    cms::cuda::free_device(dev, outerX);
    cms::cuda::free_device(dev, outerHighEdgeX);
    cms::cuda::free_device(dev, nMemoryLocations);
}


void SDL::miniDoublets::freeMemory(cudaStream_t stream)
{
    cudaFree(anchorHitIndices);
    cudaFree(moduleIndices);
    cudaFree(nMDs);
    cudaFree(totOccupancyMDs);
    cudaFree(dphichanges);
    cudaFree(anchorX);
    cudaFree(anchorHighEdgeX);
    cudaFree(outerX);
    cudaFree(outerHighEdgeX);
    cudaFree(nMemoryLocations);
}

void SDL::printMD(struct miniDoublets& mdsInGPU, struct hits& hitsInGPU, SDL::modules& modulesInGPU, unsigned int mdIndex)
{
    std::cout<<std::endl;
    std::cout << "dz " << mdsInGPU.dzs[mdIndex] << std::endl;
    std::cout << "dphi " << mdsInGPU.dphis[mdIndex] << std::endl;
    std::cout << "dphinoshift " << mdsInGPU.noShiftedDphis[mdIndex] << std::endl;
    std::cout << "dphichange " << mdsInGPU.dphichanges[mdIndex] << std::endl;
    std::cout << "dphichangenoshift " << mdsInGPU.noShiftedDphiChanges[mdIndex] << std::endl;
    std::cout << std::endl;
    std::cout << "Anchor Hit " << std::endl;
    std::cout << "------------------------------" << std::endl;
    unsigned int lowerHitIndex = mdsInGPU.anchorHitIndices[mdIndex];
    unsigned int upperHitIndex = mdsInGPU.outerHitIndices[mdIndex];
    {
        IndentingOStreambuf indent(std::cout);
        printHit(hitsInGPU, modulesInGPU, lowerHitIndex);
    }
    std::cout << "Non-anchor Hit " << std::endl;
    std::cout << "------------------------------" << std::endl;
    {
        IndentingOStreambuf indent(std::cout);
        printHit(hitsInGPU, modulesInGPU, upperHitIndex);
    }
}

__global__ void SDL::addMiniDoubletRangesToEventExplicit(struct modules& modulesInGPU, struct miniDoublets& mdsInGPU, struct objectRanges& rangesInGPU,struct hits& hitsInGPU)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int np = gridDim.x * blockDim.x;
    for(uint16_t i = gid; i < *modulesInGPU.nLowerModules; i+= np)
    {
        if(mdsInGPU.nMDs[i] == 0 or hitsInGPU.hitRanges[i * 2] == -1)
        {
            rangesInGPU.mdRanges[i * 2] = -1;
            rangesInGPU.mdRanges[i * 2 + 1] = -1;
        }
        else
        {
            rangesInGPU.mdRanges[i * 2] = rangesInGPU.miniDoubletModuleIndices[i] ;
            rangesInGPU.mdRanges[i * 2 + 1] = rangesInGPU.miniDoubletModuleIndices[i] + mdsInGPU.nMDs[i] - 1;
        }
    }
}
