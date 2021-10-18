# include "Kernels.cuh"

__global__ void createMiniDoubletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU)
{
    int lowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    //int lowerHitIndex = blockIdx.y * blockDim.y + threadIdx.y;
    //int upperHitIndex = blockIdx.z * blockDim.z + threadIdx.z;
    if(lowerModuleArrayIndex >= (*modulesInGPU.nLowerModules)) return; //extra precaution

    int lowerModuleIndex = modulesInGPU.lowerModuleIndices[lowerModuleArrayIndex];
    int upperModuleIndex = modulesInGPU.partnerModuleIndex(lowerModuleIndex);

    if(modulesInGPU.hitRanges[lowerModuleIndex * 2] == -1) return;
    if(modulesInGPU.hitRanges[upperModuleIndex * 2] == -1) return;
    unsigned int nLowerHits = modulesInGPU.hitRanges[lowerModuleIndex * 2 + 1] - modulesInGPU.hitRanges[lowerModuleIndex * 2] + 1;
    unsigned int nUpperHits = modulesInGPU.hitRanges[upperModuleIndex * 2 + 1] - modulesInGPU.hitRanges[upperModuleIndex * 2] + 1;

#ifdef NEWGRID_MD
    int lowerHitIndex =  (blockIdx.y * blockDim.y + threadIdx.y) / nUpperHits;
    int upperHitIndex =  (blockIdx.y * blockDim.y + threadIdx.y) % nUpperHits;
#else
    int lowerHitIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int upperHitIndex = blockIdx.z * blockDim.z + threadIdx.z;
#endif

    //consider assigining a dummy computation function for these
    if(lowerHitIndex >= nLowerHits) return;
    if(upperHitIndex >= nUpperHits) return;

    unsigned int lowerHitArrayIndex = modulesInGPU.hitRanges[lowerModuleIndex * 2] + lowerHitIndex;
    unsigned int upperHitArrayIndex = modulesInGPU.hitRanges[upperModuleIndex * 2] + upperHitIndex;

    float dz, drt, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange;

#ifdef CUT_VALUE_DEBUG
    float dzCut, drtCut, miniCut;
    bool success = runMiniDoubletDefaultAlgo(modulesInGPU, hitsInGPU, lowerModuleIndex, lowerHitArrayIndex, upperHitArrayIndex, dz,  drt, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, dzCut, drtCut, miniCut);
#else
    bool success = runMiniDoubletDefaultAlgo(modulesInGPU, hitsInGPU, lowerModuleIndex, lowerHitArrayIndex, upperHitArrayIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange);
#endif

    if(success)
    {
        unsigned int mdModuleIndex = atomicAdd(&mdsInGPU.nMDs[lowerModuleIndex],1);
        if(mdModuleIndex >= N_MAX_MD_PER_MODULES)
        {
            #ifdef Warnings
            if(mdModuleIndex == N_MAX_MD_PER_MODULES)
                printf("Mini-doublet excess alert! Module index =  %d\n",lowerModuleIndex);
            #endif
        }
        else
        {
            unsigned int mdIndex = lowerModuleIndex * N_MAX_MD_PER_MODULES + mdModuleIndex;
#ifdef CUT_VALUE_DEBUG
            addMDToMemory(mdsInGPU,hitsInGPU, modulesInGPU, lowerHitArrayIndex, upperHitArrayIndex, lowerModuleIndex, dz,drt, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, dzCut, drtCut, miniCut, mdIndex);
#else
        addMDToMemory(mdsInGPU,hitsInGPU, modulesInGPU, lowerHitArrayIndex, upperHitArrayIndex, lowerModuleIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, mdIndex);
#endif

        }

    }
}
__global__ void createSegmentsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU)
{
#ifdef NEWGRID_Seg
    int innerLowerModuleArrayIdx = blockIdx.z * blockDim.z + threadIdx.z;
    int outerLowerModuleArrayIdx = blockIdx.y * blockDim.y + threadIdx.y;
#else
    int xAxisIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int innerMDArrayIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int outerMDArrayIdx = blockIdx.z * blockDim.z + threadIdx.z;

    int innerLowerModuleArrayIdx = xAxisIdx/MAX_CONNECTED_MODULES;
    int outerLowerModuleArrayIdx = xAxisIdx % MAX_CONNECTED_MODULES; //need this index from the connected module array
#endif
    if(innerLowerModuleArrayIdx >= *modulesInGPU.nLowerModules) return;

    unsigned int innerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerLowerModuleArrayIdx];

    unsigned int nConnectedModules = modulesInGPU.nConnectedModules[innerLowerModuleIndex];

    if(outerLowerModuleArrayIdx >= nConnectedModules) return;

    unsigned int outerLowerModuleIndex = modulesInGPU.moduleMap[innerLowerModuleIndex * MAX_CONNECTED_MODULES + outerLowerModuleArrayIdx];

    unsigned int nInnerMDs = mdsInGPU.nMDs[innerLowerModuleIndex] > N_MAX_MD_PER_MODULES ? N_MAX_MD_PER_MODULES : mdsInGPU.nMDs[innerLowerModuleIndex];
    unsigned int nOuterMDs = mdsInGPU.nMDs[outerLowerModuleIndex] > N_MAX_MD_PER_MODULES ? N_MAX_MD_PER_MODULES : mdsInGPU.nMDs[outerLowerModuleIndex];

#ifdef NEWGRID_Seg
    if (nInnerMDs*nOuterMDs == 0) return;
    int innerMDArrayIdx = (blockIdx.x * blockDim.x + threadIdx.x) / nOuterMDs;
    int outerMDArrayIdx = (blockIdx.x * blockDim.x + threadIdx.x) % nOuterMDs;
#endif

    if(innerMDArrayIdx >= nInnerMDs) return;
    if(outerMDArrayIdx >= nOuterMDs) return;

    unsigned int innerMDIndex = modulesInGPU.mdRanges[innerLowerModuleIndex * 2] + innerMDArrayIdx;
    unsigned int outerMDIndex = modulesInGPU.mdRanges[outerLowerModuleIndex * 2] + outerMDArrayIdx;

    float zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD;

    unsigned int innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex;

    dPhiMin = 0;
    dPhiMax = 0;
    dPhiChangeMin = 0;
    dPhiChangeMax = 0;
    float zLo, zHi, rtLo, rtHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold, dAlphaInnerMDOuterMDThreshold;

    bool success = runSegmentDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, zLo, zHi, rtLo, rtHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold,
            dAlphaInnerMDOuterMDThreshold, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex);

    if(success)
    {
        unsigned int segmentModuleIdx = atomicAdd(&segmentsInGPU.nSegments[innerLowerModuleIndex],1);
        if(segmentModuleIdx >= N_MAX_SEGMENTS_PER_MODULE)
        {
            #ifdef Warnings
            if(segmentModuleIdx == N_MAX_SEGMENTS_PER_MODULE)
                printf("Segment excess alert! Module index = %d\n",innerLowerModuleIndex);
            #endif
        }
        else
        {
            unsigned int segmentIdx = innerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + segmentModuleIdx;
#ifdef CUT_VALUE_DEBUG
            addSegmentToMemory(segmentsInGPU,innerMDIndex, outerMDIndex,innerLowerModuleIndex, outerLowerModuleIndex, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, zIn, zOut, rtIn, rtOut, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, zLo, zHi, rtLo, rtHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold,
                dAlphaInnerMDOuterMDThreshold, segmentIdx);
#else
            addSegmentToMemory(segmentsInGPU,innerMDIndex, outerMDIndex,innerLowerModuleIndex, outerLowerModuleIndex, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, segmentIdx);
#endif

        }
    }
}
#ifdef NEWGRID_Tracklet
__global__ void createTrackletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int *index_gpu)
{
  //int innerInnerLowerModuleArrayIndex = blockIdx.z * blockDim.z + threadIdx.z;
  int innerInnerLowerModuleArrayIndex = index_gpu[blockIdx.z * blockDim.z + threadIdx.z];
  if(innerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
  unsigned int innerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerInnerLowerModuleArrayIndex];
  unsigned int nInnerSegments = segmentsInGPU.nSegments[innerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[innerInnerLowerModuleIndex];

  if(nInnerSegments == 0) return;

  int outerInnerLowerModuleArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;
  int innerSegmentArrayIndex = (blockIdx.x * blockDim.x + threadIdx.x) % nInnerSegments;
  int outerSegmentArrayIndex = (blockIdx.x * blockDim.x + threadIdx.x) / nInnerSegments;

  if(innerSegmentArrayIndex >= nInnerSegments) return;

  //outer inner lower module array indices should be obtained from the partner module of the inner segment's outer lower module
  unsigned int innerSegmentIndex = innerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + innerSegmentArrayIndex;

  unsigned int innerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[innerSegmentIndex];

  //number of possible outer segment inner MD lower modules
  unsigned int nOuterInnerLowerModules = modulesInGPU.nConnectedModules[innerOuterLowerModuleIndex];
  if(outerInnerLowerModuleArrayIndex >= nOuterInnerLowerModules) return;

  unsigned int outerInnerLowerModuleIndex = modulesInGPU.moduleMap[innerOuterLowerModuleIndex * MAX_CONNECTED_MODULES + outerInnerLowerModuleArrayIndex];

  unsigned int nOuterSegments = segmentsInGPU.nSegments[outerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[outerInnerLowerModuleIndex];
  if(outerSegmentArrayIndex >= nOuterSegments) return;

  unsigned int outerSegmentIndex = outerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + outerSegmentArrayIndex;

  //for completeness - outerOuterLowerModuleIndex
  unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];

  //with both segment indices obtained, run the tracklet algorithm
  float zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta;

    float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;
    bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, innerOuterLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, N_MAX_SEGMENTS_PER_MODULE); //might want to send the other two module indices and the anchor hits also to save memory accesses


  if(success)
    {
      unsigned int trackletModuleIndex = atomicAdd(&trackletsInGPU.nTracklets[innerInnerLowerModuleArrayIndex],1);
      if(trackletModuleIndex >= N_MAX_TRACKLETS_PER_MODULE)
      {
          #ifdef Warnings
          if(trackletModuleIndex == N_MAX_TRACKLETS_PER_MODULE)
              printf("Tracklet excess alert! Module index = %d\n",innerInnerLowerModuleIndex);
          #endif
      }
      else
      {
          unsigned int trackletIndex = innerInnerLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + trackletModuleIndex;
#ifdef CUT_VALUE_DEBUG
          addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, trackletIndex);

#else
          addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,trackletIndex);

#endif

      }
    }
}
#endif
//#else
//__global__ void createTrackletsFromInnerInnerLowerModule(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, unsigned int innerInnerLowerModuleIndex, unsigned int nInnerSegments, unsigned int innerInnerLowerModuleArrayIndex)
//{
//    int outerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
//    int innerSegmentArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;
//    int outerSegmentArrayIndex = blockIdx.z * blockDim.z + threadIdx.z;
//
//    if(innerSegmentArrayIndex >= nInnerSegments) return;
//        //outer inner lower module array indices should be obtained from the partner module of the inner segment's outer lower module
//    unsigned int innerSegmentIndex = innerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + innerSegmentArrayIndex;
//
//
//    unsigned int innerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[innerSegmentIndex];
//
//    //number of possible outer segment inner MD lower modules
//    unsigned int nOuterInnerLowerModules = modulesInGPU.nConnectedModules[innerOuterLowerModuleIndex];
//    if(outerInnerLowerModuleArrayIndex >= nOuterInnerLowerModules) return;
//
//    unsigned int outerInnerLowerModuleIndex = modulesInGPU.moduleMap[innerOuterLowerModuleIndex * MAX_CONNECTED_MODULES + outerInnerLowerModuleArrayIndex];
//
//    unsigned int nOuterSegments = segmentsInGPU.nSegments[outerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[outerInnerLowerModuleIndex];
//    if(outerSegmentArrayIndex >= nOuterSegments) return;
//
//    unsigned int outerSegmentIndex = outerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + outerSegmentArrayIndex;
//
//    //for completeness - outerOuterLowerModuleIndex
//    unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];
//
//    //with both segment indices obtained, run the tracklet algorithm
//
//    float zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta;
//
//    float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;
//    bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, innerOuterLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, N_MAX_SEGMENTS_PER_MODULE); //might want to send the other two module indices and the anchor hits also to save memory accesses
//
//   if(success)
//   {
//        unsigned int trackletModuleIndex = atomicAdd(&trackletsInGPU.nTracklets[innerInnerLowerModuleArrayIndex],1);
//        if(trackletModuleIndex >= N_MAX_TRACKLETS_PER_MODULE)
//        {
//            #ifdef Warnings
//            if(trackletModuleIndex == N_MAX_TRACKLETS_PER_MODULE)
//                printf("Tracklet excess alert! Module index = %d\n",innerInnerLowerModuleIndex);
//            #endif
//        }
//        else
//        {
//            unsigned int trackletIndex = innerInnerLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + trackletModuleIndex;
//#ifdef CUT_VALUE_DEBUG
//            addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, trackletIndex);
//
//#else
//            addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,trackletIndex);
//
//#endif
//        }
//   }
//
//
//
//}
//
//__global__ void createTrackletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU)
//{
//  int innerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
//  if(innerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
//  unsigned int innerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerInnerLowerModuleArrayIndex];
//  unsigned int nInnerSegments = segmentsInGPU.nSegments[innerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[innerInnerLowerModuleIndex];
//  if(nInnerSegments == 0) return;
//
//  dim3 nThreads(1,16,16);
//  dim3 nBlocks(MAX_CONNECTED_MODULES % nThreads.x  == 0 ? MAX_CONNECTED_MODULES / nThreads.x : MAX_CONNECTED_MODULES / nThreads.x + 1 ,nInnerSegments % nThreads.y == 0 ? nInnerSegments/nThreads.y : nInnerSegments/nThreads.y + 1,N_MAX_SEGMENTS_PER_MODULE % nThreads.z == 0 ? N_MAX_SEGMENTS_PER_MODULE/nThreads.z : N_MAX_SEGMENTS_PER_MODULE/nThreads.z + 1);
//
//  createTrackletsFromInnerInnerLowerModule<<<nBlocks,nThreads>>>(modulesInGPU,hitsInGPU,mdsInGPU,segmentsInGPU,trackletsInGPU,innerInnerLowerModuleIndex,nInnerSegments,innerInnerLowerModuleArrayIndex);
//
//}
//#endif
#ifdef NEWGRID_Tracklet
__global__ void createTrackletsFromTriplets(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::tracklets& trackletsInGPU,unsigned int *threadIdx_gpu, unsigned int *threadIdx_gpu_offset)
{

  int innerInnerLowerModuleArrayIndex = threadIdx_gpu[blockIdx.y * blockDim.y + threadIdx.y];
  if(innerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
  unsigned int nTriplets = tripletsInGPU.nTriplets[innerInnerLowerModuleArrayIndex] > N_MAX_TRIPLETS_PER_MODULE ? N_MAX_TRIPLETS_PER_MODULE : tripletsInGPU.nTriplets[innerInnerLowerModuleArrayIndex];

  if(nTriplets == 0) return;
  int innerTripletArrayIndex = threadIdx_gpu_offset[blockIdx.y * blockDim.y + threadIdx.y];
  int outerTripletArrayIndex = (blockIdx.x * blockDim.x + threadIdx.x);

//////////////////////////////////////////////////////////
  if(innerTripletArrayIndex >= nTriplets) return;

  //outer inner lower module array indices should be obtained from the partner module of the inner Triplet's outer lower module
  unsigned int innerTripletIndex = innerInnerLowerModuleArrayIndex * N_MAX_TRIPLETS_PER_MODULE + innerTripletArrayIndex;
  unsigned int outerInnerInnerLowerModuleIndex = modulesInGPU.reverseLookupLowerModuleIndices[tripletsInGPU.lowerModuleIndices[3 * innerTripletIndex + 1]];//same as innerOuterInnerLowerModuleIndex
        if(outerTripletArrayIndex < fminf(tripletsInGPU.nTriplets[outerInnerInnerLowerModuleIndex],N_MAX_TRIPLETS_PER_MODULE))
        {
            unsigned int outerTripletIndex = outerInnerInnerLowerModuleIndex * N_MAX_TRIPLETS_PER_MODULE + outerTripletArrayIndex;
            unsigned int innerOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex + 1];
            unsigned int outerInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex];

            if(innerOuterSegmentIndex == outerInnerSegmentIndex)
            {
              unsigned int innerSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex];
              unsigned int outerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex + 1];
              unsigned int innerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[innerSegmentIndex];
              unsigned int outerInnerLowerModuleIndex = segmentsInGPU.innerLowerModuleIndices[outerSegmentIndex];
              unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];
              float zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta;
              unsigned int innerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerInnerLowerModuleArrayIndex];

              float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;
              bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, innerOuterLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, N_MAX_SEGMENTS_PER_MODULE); //might want to send the other two module indices and the anchor hits also to save memory accesses

              if(success)
              {
                   unsigned int trackletModuleIndex = atomicAdd(&trackletsInGPU.nTracklets[innerInnerLowerModuleArrayIndex],1);
                   if(trackletModuleIndex >= N_MAX_TRACKLETS_PER_MODULE)
                   {
                       #ifdef Warnings
                       if(trackletModuleIndex == N_MAX_TRACKLETS_PER_MODULE)
                           printf("Tracklet excess alert! Module index = %d\n",innerInnerLowerModuleIndex);
                       #endif
                   }
                   else
                   {
                       unsigned int trackletIndex = innerInnerLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + trackletModuleIndex;
                        #ifdef CUT_VALUE_DEBUG
                       addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, trackletIndex);
                        #else
                       addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,trackletIndex);
                        #endif
                   }
              }
            }
        }
}
#else
__global__ void createTrackletsFromTriplets(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::tracklets& trackletsInGPU/*,unsigned int *index_gpu*/)
{
  int innerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if(innerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
  unsigned int nTriplets = tripletsInGPU.nTriplets[innerInnerLowerModuleArrayIndex] > N_MAX_TRIPLETS_PER_MODULE ? N_MAX_TRIPLETS_PER_MODULE : tripletsInGPU.nTriplets[innerInnerLowerModuleArrayIndex];

  if(nTriplets == 0) return;
    dim3 nThreads(16,16,1);
    dim3 nBlocks(nTriplets % nThreads.x == 0 ? nTriplets / nThreads.x : nTriplets / nThreads.x + 1, N_MAX_TRIPLETS_PER_MODULE % nThreads.y == 0 ? N_MAX_TRIPLETS_PER_MODULE / nThreads.y : N_MAX_TRIPLETS_PER_MODULE / nThreads.y + 1, 1);
    createTrackletsFromTripletsP2<<<nBlocks,nThreads>>>(modulesInGPU,hitsInGPU,mdsInGPU,segmentsInGPU,tripletsInGPU,trackletsInGPU,innerInnerLowerModuleArrayIndex,nTriplets);

}
__global__ void createTrackletsFromTripletsP2(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::tracklets& trackletsInGPU/*,unsigned int *index_gpu*/,unsigned int innerInnerLowerModuleArrayIndex, unsigned int nTriplets)
{
  int innerTripletArrayIndex = (blockIdx.x * blockDim.x + threadIdx.x);// % nTriplets;
  int outerTripletArrayIndex = (blockIdx.y * blockDim.y + threadIdx.y);// / nTriplets;
  if(innerTripletArrayIndex >= nTriplets) return;

  //outer inner lower module array indices should be obtained from the partner module of the inner Triplet's outer lower module
  unsigned int innerTripletIndex = innerInnerLowerModuleArrayIndex * N_MAX_TRIPLETS_PER_MODULE + innerTripletArrayIndex;
  unsigned int outerInnerInnerLowerModuleIndex = modulesInGPU.reverseLookupLowerModuleIndices[tripletsInGPU.lowerModuleIndices[3 * innerTripletIndex + 1]];//same as innerOuterInnerLowerModuleIndex
        if(outerTripletArrayIndex < fminf(tripletsInGPU.nTriplets[outerInnerInnerLowerModuleIndex],N_MAX_TRIPLETS_PER_MODULE))
        {
            unsigned int outerTripletIndex = outerInnerInnerLowerModuleIndex * N_MAX_TRIPLETS_PER_MODULE + outerTripletArrayIndex;
            unsigned int innerOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex + 1];
            unsigned int outerInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex];

            if(innerOuterSegmentIndex == outerInnerSegmentIndex)
            {
              unsigned int innerSegmentIndex = tripletsInGPU.segmentIndices[2 * innerTripletIndex];
              unsigned int outerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerTripletIndex + 1];
              unsigned int innerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[innerSegmentIndex];
              unsigned int outerInnerLowerModuleIndex = segmentsInGPU.innerLowerModuleIndices[outerSegmentIndex];
              unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];
              float zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta;
              unsigned int innerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerInnerLowerModuleArrayIndex];

              float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;
              bool success = runTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, innerOuterLowerModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, N_MAX_SEGMENTS_PER_MODULE); //might want to send the other two module indices and the anchor hits also to save memory accesses

              if(success)
              {
                   unsigned int trackletModuleIndex = atomicAdd(&trackletsInGPU.nTracklets[innerInnerLowerModuleArrayIndex],1);
                   if(trackletModuleIndex >= N_MAX_TRACKLETS_PER_MODULE)
                   {
                       #ifdef Warnings
                       if(trackletModuleIndex == N_MAX_TRACKLETS_PER_MODULE)
                           printf("Tracklet excess alert! Module index = %d\n",innerInnerLowerModuleIndex);
                       #endif
                   }
                   else
                   {
                       unsigned int trackletIndex = innerInnerLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + trackletModuleIndex;
                        #ifdef CUT_VALUE_DEBUG
                       addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, trackletIndex);
                        #else
                       addTrackletToMemory(trackletsInGPU,innerSegmentIndex,outerSegmentIndex,innerInnerLowerModuleIndex,innerOuterLowerModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,trackletIndex);
                        #endif
                   }
              }
            }
        }
}
#endif
//__global__ void createPixelTrackletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTracklets& pixelTrackletsInGPU, unsigned int* threadIdx_gpu, unsigned int *threadIdx_gpu_offset)
//{
//  int outerInnerLowerModuleArrayIndex = threadIdx_gpu[blockIdx.y * blockDim.y + threadIdx.y];
//  if(outerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
//
//  unsigned int outerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[outerInnerLowerModuleArrayIndex];
//  unsigned int pixelModuleIndex = *modulesInGPU.nModules - 1; //last dude
//  unsigned int pixelLowerModuleArrayIndex = modulesInGPU.reverseLookupLowerModuleIndices[pixelModuleIndex]; //should be the same as nLowerModules
//  unsigned int nInnerSegments = segmentsInGPU.nSegments[pixelModuleIndex] > N_MAX_PIXEL_SEGMENTS_PER_MODULE ? N_MAX_PIXEL_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[pixelModuleIndex];
//  unsigned int nOuterSegments = segmentsInGPU.nSegments[outerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[outerInnerLowerModuleIndex];
//  if(nOuterSegments == 0) return;
//  if(nInnerSegments == 0) return;
//  if(modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::TwoS) return; //REMOVES 2S-2S
//
//  int innerSegmentArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
//  int outerSegmentArrayIndex = threadIdx_gpu_offset[blockIdx.y * blockDim.y + threadIdx.y];
//  if(innerSegmentArrayIndex >= nInnerSegments) return;
//  if(outerSegmentArrayIndex >= nOuterSegments) return;
//  unsigned int innerSegmentIndex = pixelModuleIndex * N_MAX_SEGMENTS_PER_MODULE + innerSegmentArrayIndex;
//  unsigned int outerSegmentIndex = outerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + outerSegmentArrayIndex;
//  unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];
//  if(modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS) return; //REMOVES PS-2S
//  float zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta;
//
//  float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;
//  bool success = runPixelTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, pixelModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, N_MAX_SEGMENTS_PER_MODULE); //might want to send the other two module indices and the anchor hits also to save memory accesses
//  if(success)
//    {
//      unsigned int trackletModuleIndex = atomicAdd(pixelTrackletsInGPU.nPixelTracklets, 1);
//      if(trackletModuleIndex >= N_MAX_PIXEL_TRACKLETS_PER_MODULE)
//        {
//            #ifdef Warnings
//	  if(trackletModuleIndex == N_MAX_PIXEL_TRACKLETS_PER_MODULE)
//	    printf("Pixel Tracklet excess alert! Module index = %d\n",pixelModuleIndex);
//            #endif
//        }
//      else
//        {
//	  unsigned int trackletIndex = trackletModuleIndex;
//#ifdef CUT_VALUE_DEBUG
//	  addPixelTrackletToMemory(pixelTrackletsInGPU,innerSegmentIndex,outerSegmentIndex,pixelModuleIndex,pixelModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, trackletIndex);
//#else
//	  addPixelTrackletToMemory(pixelTrackletsInGPU,innerSegmentIndex,outerSegmentIndex,pixelModuleIndex,pixelModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,trackletIndex);
//#endif
//        }
//    }
//}
__global__ void createPixelTrackletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTracklets& pixelTrackletsInGPU, unsigned int* connectedPixelSize, unsigned int* connectedPixelIndex,unsigned int nInnerSegs,unsigned int* seg_pix_gpu, unsigned int* seg_pix_gpu_offset, unsigned int totalSegs)
{
  //newgrid with map
  unsigned int offsetIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if(offsetIndex >= totalSegs) return;

  int segmentArrayIndex = seg_pix_gpu_offset[offsetIndex];
  int pixelArrayIndex = seg_pix_gpu[offsetIndex];
  if(pixelArrayIndex >= nInnerSegs) return;// don't exceed # of pLS
  if( segmentArrayIndex >= connectedPixelSize[pixelArrayIndex]) return; // don't exceed # connected segment modules for this pixel

  unsigned int outerInnerLowerModuleArrayIndex;// This will be the index of the module that connects to this pixel.
    unsigned int temp = connectedPixelIndex[pixelArrayIndex]+segmentArrayIndex; //gets module index for segment
    outerInnerLowerModuleArrayIndex = modulesInGPU.connectedPixels[temp]; //gets module index for segment
  if(outerInnerLowerModuleArrayIndex >= *modulesInGPU.nModules - 1) return;
  unsigned int outerInnerLowerModuleIndex = /*modulesInGPU.lowerModuleIndices[*/outerInnerLowerModuleArrayIndex;//];

  unsigned int pixelModuleIndex = *modulesInGPU.nModules - 1; //last dude
  unsigned int nOuterSegments = segmentsInGPU.nSegments[outerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[outerInnerLowerModuleIndex];
  if(nOuterSegments == 0) return;
  if(modulesInGPU.moduleType[outerInnerLowerModuleIndex] == SDL::TwoS) return; //REMOVES 2S-2S

//  int outerSegmentArrayIndex = blockIdx.z * blockDim.z + threadIdx.z;
  int outerSegmentArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;
  if(outerSegmentArrayIndex >= nOuterSegments) return;
  unsigned int innerSegmentIndex = pixelModuleIndex * N_MAX_SEGMENTS_PER_MODULE + pixelArrayIndex;
  unsigned int outerSegmentIndex = outerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + outerSegmentArrayIndex;
  unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];
  if(modulesInGPU.moduleType[outerOuterLowerModuleIndex] == SDL::TwoS) return; //REMOVES PS-2S
  float zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta;

  float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;
  bool success = runPixelTrackletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, pixelModuleIndex, pixelModuleIndex, outerInnerLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, N_MAX_SEGMENTS_PER_MODULE); //might want to send the other two module indices and the anchor hits also to save memory accesses
  if(success)
    {
      unsigned int trackletModuleIndex = atomicAdd(pixelTrackletsInGPU.nPixelTracklets, 1);
      if(trackletModuleIndex >= N_MAX_PIXEL_TRACKLETS_PER_MODULE)
        {
            #ifdef Warnings
	  if(trackletModuleIndex == N_MAX_PIXEL_TRACKLETS_PER_MODULE)
	    printf("Pixel Tracklet excess alert! Module index = %d\n",pixelModuleIndex);
            #endif
        }
      else
        {
	  unsigned int trackletIndex = trackletModuleIndex;
#ifdef CUT_VALUE_DEBUG
	  addPixelTrackletToMemory(pixelTrackletsInGPU,innerSegmentIndex,outerSegmentIndex,pixelModuleIndex,pixelModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, trackletIndex);
#else
	  addPixelTrackletToMemory(pixelTrackletsInGPU,innerSegmentIndex,outerSegmentIndex,pixelModuleIndex,pixelModuleIndex,outerInnerLowerModuleIndex,outerOuterLowerModuleIndex,zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut,pt_beta,trackletIndex);
#endif
        }
    }
}


#ifdef NEWGRID_Trips
__global__ void createTripletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, unsigned int *index_gpu)
{
  int innerInnerLowerModuleArrayIndex = index_gpu[blockIdx.z * blockDim.z + threadIdx.z];
  if(innerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;

  unsigned int innerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerInnerLowerModuleArrayIndex];
  unsigned int nConnectedModules = modulesInGPU.nConnectedModules[innerInnerLowerModuleIndex];
  if(nConnectedModules == 0) return;

  unsigned int nInnerSegments = segmentsInGPU.nSegments[innerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[innerInnerLowerModuleIndex];

  int innerSegmentArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;
  int outerSegmentArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if(innerSegmentArrayIndex >= nInnerSegments) return;

  unsigned int innerSegmentIndex = innerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + innerSegmentArrayIndex;

  //middle lower module - outer lower module of inner segment
  unsigned int middleLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[innerSegmentIndex];

  unsigned int nOuterSegments = segmentsInGPU.nSegments[middleLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[middleLowerModuleIndex];
  if(outerSegmentArrayIndex >= nOuterSegments) return;

  unsigned int outerSegmentIndex = middleLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + outerSegmentArrayIndex;
  unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];

  float zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut, pt_beta;
  float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;

    bool success = runTripletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);

  if(success)
    {
      unsigned int tripletModuleIndex = atomicAdd(&tripletsInGPU.nTriplets[innerInnerLowerModuleArrayIndex], 1);
      if(tripletModuleIndex >= N_MAX_TRIPLETS_PER_MODULE)
      {
          #ifdef Warnings
          if(tripletModuleIndex == N_MAX_TRIPLETS_PER_MODULE)
              printf("Triplet excess alert! Module index = %d\n",innerInnerLowerModuleIndex);
          #endif
      }
      unsigned int tripletIndex = innerInnerLowerModuleArrayIndex * N_MAX_TRIPLETS_PER_MODULE + tripletModuleIndex;
#ifdef CUT_VALUE_DEBUG

        addTripletToMemory(tripletsInGPU, innerSegmentIndex, outerSegmentIndex, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut,pt_beta, zLo,zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, tripletIndex);

#else
      addTripletToMemory(tripletsInGPU, innerSegmentIndex, outerSegmentIndex, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, betaIn, betaOut, pt_beta, tripletIndex);
#endif
    }
}
#endif

__device__ inline int checkHitspT5(unsigned int ix, unsigned int jx,struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::hits& hitsInGPU)
{
    int phits1[4] = {-1,-1,-1,-1};
    int phits2[4] = {-1,-1,-1,-1};
    phits1[0] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*ix]]];
    phits1[1] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*ix+1]]];
    phits1[2] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*ix]+1]];
    phits1[3] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*ix+1]+1]];

    phits2[0] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*jx]]];
    phits2[1] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*jx+1]]];
    phits2[2] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*jx]+1]];
    phits2[3] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*jx+1]+1]];

    int npMatched =0;

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
__global__ void addT5asTrackCandidateInGPU(struct SDL::modules& modulesInGPU,struct SDL::quintuplets& quintupletsInGPU,struct SDL::trackCandidates& trackCandidatesInGPU,struct SDL::pixelQuintuplets& pixelQuintupletsInGPU,struct SDL::pixelTriplets& pixelTripletsInGPU)
{

    int innerInnerInnerLowerModuleArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(innerInnerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules or modulesInGPU.quintupletModuleIndices[innerInnerInnerLowerModuleArrayIndex] == -1) return;
    unsigned int nQuints = quintupletsInGPU.nQuintuplets[innerInnerInnerLowerModuleArrayIndex];
    if (nQuints > N_MAX_QUINTUPLETS_PER_MODULE)
    {
        nQuints = N_MAX_QUINTUPLETS_PER_MODULE;
    }
    int innerObjectArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if(innerObjectArrayIndex >= nQuints) return;
    int quintupletIndex = modulesInGPU.quintupletModuleIndices[innerInnerInnerLowerModuleArrayIndex] + innerObjectArrayIndex;

    //don't add duplicate T5s or T5s that are accounted in pT5s
    if(quintupletsInGPU.isDup[quintupletIndex] or quintupletsInGPU.partOfPT5[quintupletIndex])
    {
        return;
    }
#ifdef Crossclean_T5
    unsigned int loop_bound = *pixelQuintupletsInGPU.nPixelQuintuplets;
    if (loop_bound < *pixelTripletsInGPU.nPixelTriplets)
    {
        loop_bound = *pixelTripletsInGPU.nPixelTriplets;
    }
    //cross cleaning step
    float eta1 = quintupletsInGPU.eta[quintupletIndex];
    float phi1 = quintupletsInGPU.phi[quintupletIndex];

    for (unsigned int jx=0; jx<loop_bound; jx++)
    {
        if(jx < *pixelQuintupletsInGPU.nPixelQuintuplets)
        {
            float eta2 = pixelQuintupletsInGPU.eta[jx];
            float phi2 = pixelQuintupletsInGPU.phi[jx];
            float dEta = abs(eta1-eta2);
            float dPhi = abs(phi1-phi2);
            if(dPhi > M_PI){dPhi = dPhi - 2*M_PI;}
            float dR2 = dEta*dEta + dPhi*dPhi;
            //printf("dR2: %f\n",dR2);
            if(dR2 < 1e-3) return;
        }
        if(jx < *pixelTripletsInGPU.nPixelTriplets)
        {
            float eta2 = pixelTripletsInGPU.eta[jx];
            float phi2 = pixelTripletsInGPU.phi[jx];
            float dEta = abs(eta1-eta2);
            float dPhi = abs(phi1-phi2);
            if(dPhi > M_PI){dPhi = dPhi - 2*M_PI;}
            float dR2 = dEta*dEta + dPhi*dPhi;
            if(dR2 < 1e-3) return;
        }
    }
#endif
    unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[innerInnerInnerLowerModuleArrayIndex],1);
    atomicAdd(&trackCandidatesInGPU.nTrackCandidatesT5[innerInnerInnerLowerModuleArrayIndex],1);
    unsigned int trackCandidateIdx = modulesInGPU.trackCandidateModuleIndices[innerInnerInnerLowerModuleArrayIndex] + trackCandidateModuleIdx;
    addTrackCandidateToMemory(trackCandidatesInGPU, 4/*track candidate type T5=4*/, quintupletIndex, quintupletIndex, trackCandidateIdx);
}

__global__ void addpT2asTrackCandidateInGPU(struct SDL::modules& modulesInGPU,struct SDL::pixelTracklets& pixelTrackletsInGPU,struct SDL::trackCandidates& trackCandidatesInGPU)
{
    int pixelTrackletArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pixelLowerModuleArrayIndex = *modulesInGPU.nLowerModules;
    unsigned int nPixelTracklets = *pixelTrackletsInGPU.nPixelTracklets;
    if(pixelTrackletArrayIndex >= nPixelTracklets) return;
    int pixelTrackletIndex = pixelTrackletArrayIndex;
    unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[pixelLowerModuleArrayIndex],1);
    atomicAdd(trackCandidatesInGPU.nTrackCandidatespT2,1);
    unsigned int trackCandidateIdx = modulesInGPU.trackCandidateModuleIndices[pixelLowerModuleArrayIndex] + trackCandidateModuleIdx;
    addTrackCandidateToMemory(trackCandidatesInGPU, 3/*track candidate type pT2=3*/, pixelTrackletIndex, pixelTrackletIndex, trackCandidateIdx);
}

__global__ void addpT3asTrackCandidateInGPU(struct SDL::modules& modulesInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU,struct SDL::segments& segmentsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU)
{
    int pixelTripletArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pixelLowerModuleArrayIndex = *modulesInGPU.nLowerModules;
    unsigned int nPixelTriplets = *pixelTripletsInGPU.nPixelTriplets;
    if(pixelTripletArrayIndex >= nPixelTriplets) return;
    int pixelTripletIndex = pixelTripletArrayIndex;
    if(pixelTripletsInGPU.isDup[pixelTripletIndex])
    {
        return;
    }

#ifdef Crossclean_pT3
    //cross cleaning step
    float eta1 = pixelTripletsInGPU.eta_pix[pixelTripletIndex];
    float phi1 = pixelTripletsInGPU.phi_pix[pixelTripletIndex];
    int pixelModuleIndex = *modulesInGPU.nModules - 1;
    unsigned int prefix = pixelModuleIndex*N_MAX_SEGMENTS_PER_MODULE;
    for (unsigned int jx=0; jx<*pixelQuintupletsInGPU.nPixelQuintuplets; jx++)
    {
        unsigned int pLS_jx = pixelQuintupletsInGPU.pixelIndices[jx];
        float eta2 = segmentsInGPU.eta[pLS_jx - prefix];
        float phi2 = segmentsInGPU.phi[pLS_jx - prefix];
        float dEta = abs(eta1-eta2);
        float dPhi = abs(phi1-phi2);
        if(dPhi > M_PI){dPhi = dPhi - 2*M_PI;}
        float dR2 = dEta*dEta + dPhi*dPhi;
        if(dR2 < 1e-5) return;
    }
#endif


    unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[pixelLowerModuleArrayIndex],1);
    atomicAdd(trackCandidatesInGPU.nTrackCandidatespT3,1);
    unsigned int trackCandidateIdx = modulesInGPU.trackCandidateModuleIndices[pixelLowerModuleArrayIndex] + trackCandidateModuleIdx;
    addTrackCandidateToMemory(trackCandidatesInGPU, 5/*track candidate type pT3=5*/, pixelTripletIndex, pixelTripletIndex, trackCandidateIdx);
}

__global__ void addpLSasTrackCandidateInGPU(struct SDL::modules& modulesInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU,struct SDL::segments& segmentsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU,struct SDL::miniDoublets& mdsInGPU, struct SDL::hits& hitsInGPU, struct SDL::quintuplets& quintupletsInGPU)
{
    int pixelArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pixelLowerModuleArrayIndex = *modulesInGPU.nLowerModules;
    int pixelModuleIndex = *modulesInGPU.nModules - 1;
    unsigned int nPixels = segmentsInGPU.nSegments[pixelModuleIndex];
    if(pixelArrayIndex >= nPixels) return;

    if((!segmentsInGPU.isQuad[pixelArrayIndex]) || (segmentsInGPU.isDup[pixelArrayIndex]))
    {
        return;
    }


      //cross cleaning step

    float eta1 = segmentsInGPU.eta[pixelArrayIndex];
    float phi1 = segmentsInGPU.phi[pixelArrayIndex];
    unsigned int prefix = pixelModuleIndex*N_MAX_SEGMENTS_PER_MODULE;

    unsigned int loop_bound = *pixelQuintupletsInGPU.nPixelQuintuplets;
    if (loop_bound < *pixelTripletsInGPU.nPixelTriplets) { loop_bound = *pixelTripletsInGPU.nPixelTriplets;}
    for (unsigned int idx = 0; idx < *(modulesInGPU.nLowerModules); idx++) // "<=" because cheating to include pixel track candidate lower module
    {

        if (modulesInGPU.trackCandidateModuleIndices[idx] == -1)
        {
            continue;
        }

        unsigned int nTrackCandidates = trackCandidatesInGPU.nTrackCandidates[idx];

        if (idx == *(modulesInGPU.nLowerModules) && nTrackCandidates > N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE)
        {
            nTrackCandidates = N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE;
        }

        if (idx < *(modulesInGPU.nLowerModules) && nTrackCandidates > 50000)
        {
            nTrackCandidates = 50000;
        }
    for (unsigned int jx=0; jx<nTrackCandidates; jx++)
    {
        unsigned int trackCandidateIndex = modulesInGPU.trackCandidateModuleIndices[idx] + jx; // this line causes the issue
        short type = trackCandidatesInGPU.trackCandidateType[trackCandidateIndex];
        unsigned int innerTrackletIdx = trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex];
        if(type == 4)
        {
            unsigned int quintupletIndex = innerTrackletIdx;//trackCandidatesInGPU.objectIndices[2*jx];//T5 index
            float eta2 = quintupletsInGPU.eta[quintupletIndex];
            float phi2 = quintupletsInGPU.phi[quintupletIndex];
            float dEta = abs(eta1-eta2);
            float dPhi = abs(phi1-phi2);
            if(dPhi > M_PI){dPhi = dPhi - 2*M_PI;}
            float dR2 = dEta*dEta + dPhi*dPhi;
            if(dR2 < 1e-3) return;


        }
    }
  }

    for (unsigned int jx=0; jx<loop_bound; jx++)
    {
        if(jx < *pixelQuintupletsInGPU.nPixelQuintuplets)
        {
            if(!pixelQuintupletsInGPU.isDup[jx]){
            unsigned int pLS_jx = pixelQuintupletsInGPU.pixelIndices[jx];
            int npMatched = checkHitspT5(prefix+pixelArrayIndex,pLS_jx,mdsInGPU,segmentsInGPU,hitsInGPU);
            if(npMatched >0)
            {
                return;
            }
            float eta2 = segmentsInGPU.eta[pLS_jx - prefix];
            float phi2 = segmentsInGPU.phi[pLS_jx - prefix];
            float dEta = abs(eta1-eta2);
            float dPhi = abs(phi1-phi2);
            if(dPhi > M_PI){dPhi = dPhi - 2*M_PI;}
            float dR2 = dEta*dEta + dPhi*dPhi;
            if(dR2 < 0.000001) return;
            }
        }
        if(jx < *pixelTripletsInGPU.nPixelTriplets)
        {
            if(!pixelTripletsInGPU.isDup[jx]){
            int pLS_jx = pixelTripletsInGPU.pixelSegmentIndices[jx];
            int npMatched = checkHitspT5(prefix+pixelArrayIndex,pLS_jx,mdsInGPU,segmentsInGPU,hitsInGPU);
            if(npMatched >0)
            {
                return;
            }
            float eta2 = pixelTripletsInGPU.eta_pix[jx];
            float phi2 = pixelTripletsInGPU.phi_pix[jx];
            float dEta = abs(eta1-eta2);
            float dPhi = abs(phi1-phi2);
            if(dPhi > M_PI){dPhi = dPhi - 2*M_PI;}
            float dR2 = dEta*dEta + dPhi*dPhi;
            if(dR2 < 0.000001) return;
            }
        }
    }
    if(segmentsInGPU.score[pixelArrayIndex] > 120){return;}



    unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[pixelLowerModuleArrayIndex],1);
    atomicAdd(trackCandidatesInGPU.nTrackCandidatespLS,1);
    unsigned int trackCandidateIdx = modulesInGPU.trackCandidateModuleIndices[pixelLowerModuleArrayIndex] + trackCandidateModuleIdx;
    addTrackCandidateToMemory(trackCandidatesInGPU, 8/*track candidate type pLS=8*/, pixelArrayIndex, pixelArrayIndex, trackCandidateIdx);

}


__global__ void addpT5asTrackCandidateInGPU(struct SDL::modules& modulesInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU)
{
    int pixelQuintupletArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pixelLowerModuleArrayIndex = *modulesInGPU.nLowerModules;
    unsigned int nPixelQuintuplets = *pixelQuintupletsInGPU.nPixelQuintuplets;
    if(pixelQuintupletArrayIndex >= nPixelQuintuplets) return;
    int pixelQuintupletIndex = pixelQuintupletArrayIndex;
    if(pixelQuintupletsInGPU.isDup[pixelQuintupletIndex])
    {
        return;
    }
    unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[pixelLowerModuleArrayIndex],1);
    atomicAdd(trackCandidatesInGPU.nTrackCandidatespT5,1);
    unsigned int trackCandidateIdx = modulesInGPU.trackCandidateModuleIndices[pixelLowerModuleArrayIndex] + trackCandidateModuleIdx;


    addTrackCandidateToMemory(trackCandidatesInGPU, 7/*track candidate type pT5=7*/, pixelQuintupletsInGPU.pixelIndices[pixelQuintupletIndex], pixelQuintupletsInGPU.T5Indices[pixelQuintupletIndex], trackCandidateIdx);

}

__global__ void createPixelTrackCandidatesInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTracklets& pixelTrackletsInGPU, struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, unsigned int* threadIdx_gpu, unsigned int* threadIdx_gpu_offset)
{
    unsigned int outerInnerInnerLowerModuleArrayIndex = threadIdx_gpu[blockIdx.y * blockDim.y + threadIdx.y];
    if(outerInnerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;
    //FIXME:Cheapo module map - We care about pT4s and pTCs Only if the outerInnerInnerLowerModule is "connected" to the pixel module

    int outerInnerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[outerInnerInnerLowerModuleArrayIndex];
    if(modulesInGPU.moduleType[outerInnerInnerLowerModuleIndex] == SDL::TwoS) return;

    unsigned int pixelLowerModuleArrayIndex = *modulesInGPU.nLowerModules;

    unsigned int nPixelTracklets = *(pixelTrackletsInGPU.nPixelTracklets);
    //capping
    if(nPixelTracklets > N_MAX_PIXEL_TRACKLETS_PER_MODULE)
    {
        nPixelTracklets = N_MAX_PIXEL_TRACKLETS_PER_MODULE;
    }

    unsigned int nOuterLayerTracklets = trackletsInGPU.nTracklets[outerInnerInnerLowerModuleArrayIndex];
    if(nOuterLayerTracklets > N_MAX_TRACKLETS_PER_MODULE)
    {
        nOuterLayerTracklets = N_MAX_TRACKLETS_PER_MODULE;
    }
    unsigned int nOuterLayerTriplets = tripletsInGPU.nTriplets[outerInnerInnerLowerModuleArrayIndex];
    if(nOuterLayerTriplets > N_MAX_TRIPLETS_PER_MODULE)
    {
        nOuterLayerTriplets = N_MAX_TRIPLETS_PER_MODULE;
    }

    unsigned int nThreadsForNestedKernel = max(nOuterLayerTracklets,nOuterLayerTriplets);
    if(nThreadsForNestedKernel == 0) return;

    int pixelTrackletArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;

    int outerObjectArrayIndex = threadIdx_gpu_offset[blockIdx.y * blockDim.y+ threadIdx.y];
    if(pixelTrackletArrayIndex >= nPixelTracklets) return;

    int pixelTrackletIndex = pixelTrackletArrayIndex;
    int outerObjectIndex = 0;
    short trackCandidateType;
    bool success;

    //pT4-T4
    if(outerObjectArrayIndex < nOuterLayerTracklets)
    {
        outerObjectIndex = outerInnerInnerLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + outerObjectArrayIndex;

        //part 2 of cheapo module map : only considering tracklets with PS-PS inner segment
        if(modulesInGPU.moduleType[trackletsInGPU.lowerModuleIndices[4 * outerObjectIndex + 1]] == SDL::PS)
        {
	        success = runTrackCandidateDefaultAlgoTwoTracklets(pixelTrackletsInGPU, trackletsInGPU, tripletsInGPU, pixelTrackletIndex, outerObjectIndex, trackCandidateType);
	    if(success)
        {
	        unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[pixelLowerModuleArrayIndex],1);
	        atomicAdd(&trackCandidatesInGPU.nTrackCandidatesT4T4[pixelLowerModuleArrayIndex],1);
	        if(trackCandidateModuleIdx >= N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE)
            {
                #ifdef Warnings
    		  if(innerInnerInnerLowerModuleArrayIndex == *modulesInGPU.nLowerModules && trackCandidateModuleIdx == N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE)
                {

		            printf("Track Candidate excess alert! lower Module array index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                }
                    #endif
            }
	        else
            {
		    if(modulesInGPU.trackCandidateModuleIndices[pixelLowerModuleArrayIndex] == -1)
            {
                #ifdef Warnings
		        printf("Track candidates: no memory for pixel lower module index at %d\n",innerInnerInnerLowerModuleArrayIndex);
                #endif

            }
		  else
		    {
		        unsigned int trackCandidateIdx = modulesInGPU.trackCandidateModuleIndices[pixelLowerModuleArrayIndex] + trackCandidateModuleIdx;
		        addTrackCandidateToMemory(trackCandidatesInGPU, 5/*trackCandidateType*/, pixelTrackletIndex, outerObjectIndex, trackCandidateIdx);
                    }

                }
            }
        }
    }

    //pT4-T3
    if(outerObjectArrayIndex < nOuterLayerTriplets)
    {
        outerObjectIndex = outerInnerInnerLowerModuleArrayIndex * N_MAX_TRIPLETS_PER_MODULE + outerObjectArrayIndex;

        //part 2 of cheapo module map : only considering tracklets with PS-PS inner segment
        if(modulesInGPU.moduleType[tripletsInGPU.lowerModuleIndices[3 * outerObjectIndex + 1]] == SDL::PS)
        {
	        success = runTrackCandidateDefaultAlgoTrackletToTriplet(pixelTrackletsInGPU, trackletsInGPU, tripletsInGPU, pixelTrackletIndex, outerObjectIndex, trackCandidateType);
	        if(success)
            {
	            unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[pixelLowerModuleArrayIndex],1);
	            atomicAdd(&trackCandidatesInGPU.nTrackCandidatesT4T4[pixelLowerModuleArrayIndex],1);
	            if(trackCandidateModuleIdx >= N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE)
                {
#ifdef Warnings
		            if(innerInnerInnerLowerModuleArrayIndex == *modulesInGPU.nLowerModules && trackCandidateModuleIdx == N_MAX_PIXEL_TRACK_CANDIDATES_PER_MODULE)
                    {

		                printf("Track Candidate excess alert! lower Module array index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                    }
                    #endif
                }
	            else
                {
            		if(modulesInGPU.trackCandidateModuleIndices[pixelLowerModuleArrayIndex] == -1)
                    {
#ifdef Warnings
		                printf("Track candidates: no memory for pixel lower module index at %d\n",innerInnerInnerLowerModuleArrayIndex);
#endif
                    }
		            else
		            {
		                unsigned int trackCandidateIdx = modulesInGPU.trackCandidateModuleIndices[pixelLowerModuleArrayIndex] + trackCandidateModuleIdx;
		                addTrackCandidateToMemory(trackCandidatesInGPU, 6/*trackCandidateType*/, pixelTrackletIndex, outerObjectIndex, trackCandidateIdx);
                    }
                }
            }
        }

    }
}

__global__ void createTrackCandidatesInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::tracklets& trackletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, unsigned int* threadIdx_gpu, unsigned int *threadIdx_gpu_offset)
{
    //inner tracklet/triplet inner segment inner MD lower module
    int innerInnerInnerLowerModuleArrayIndex = threadIdx_gpu[blockIdx.y * blockDim.y + threadIdx.y];
    //hack to include pixel detector
    if(innerInnerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;

    unsigned int nTracklets = trackletsInGPU.nTracklets[innerInnerInnerLowerModuleArrayIndex];
    if(nTracklets > N_MAX_TRACKLETS_PER_MODULE)
    {
        nTracklets = N_MAX_TRACKLETS_PER_MODULE;
    }

    unsigned int nTriplets = tripletsInGPU.nTriplets[innerInnerInnerLowerModuleArrayIndex]; // should be zero for the pixels
    if(nTriplets > N_MAX_TRIPLETS_PER_MODULE)
    {
        nTriplets = N_MAX_TRIPLETS_PER_MODULE;
    }

    unsigned int temp = max(nTracklets,nTriplets);
    unsigned int MAX_OBJECTS = max(N_MAX_TRACKLETS_PER_MODULE, N_MAX_TRIPLETS_PER_MODULE);

    if(temp == 0) return;

    int innerObjectArrayIndex = threadIdx_gpu_offset[blockIdx.y * blockDim.y + threadIdx.y];
    int outerObjectArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;

    int innerObjectIndex = 0;
    int outerObjectIndex = 0;
    short trackCandidateType;
    bool success;

    //step 1 tracklet-tracklet
    if(innerObjectArrayIndex < nTracklets)
    {
        innerObjectIndex = innerInnerInnerLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + innerObjectArrayIndex;
        unsigned int outerInnerInnerLowerModuleIndex = modulesInGPU.reverseLookupLowerModuleIndices[trackletsInGPU.lowerModuleIndices[4 * innerObjectIndex + 2]];/*same as innerOuterInnerLowerModuleIndex*/

        if(outerObjectArrayIndex < fminf(trackletsInGPU.nTracklets[outerInnerInnerLowerModuleIndex],N_MAX_TRACKLETS_PER_MODULE))
        {

	        outerObjectIndex = outerInnerInnerLowerModuleIndex * N_MAX_TRACKLETS_PER_MODULE + outerObjectArrayIndex;

	        success = runTrackCandidateDefaultAlgoTwoTracklets(trackletsInGPU, tripletsInGPU, innerObjectIndex, outerObjectIndex,trackCandidateType);

	        if(success)
            {
	            unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[innerInnerInnerLowerModuleArrayIndex],1);
	            atomicAdd(&trackCandidatesInGPU.nTrackCandidatesT4T4[innerInnerInnerLowerModuleArrayIndex],1);
	            if(trackCandidateModuleIdx >= N_MAX_TRACK_CANDIDATES_PER_MODULE)
                {
#ifdef Warnings
    		        if(trackCandidateModuleIdx == N_MAX_TRACK_CANDIDATES_PER_MODULE)
                    {
		                printf("Track Candidate excess alert! lower Module array index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                    }
#endif
                }
	            else
                {
		            if(modulesInGPU.trackCandidateModuleIndices[innerInnerInnerLowerModuleArrayIndex] == -1)
                    {
                        #ifdef Warnings
		                printf("Track candidates: no memory for module at module index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                       #endif

                    }
		            else
		            {
		                unsigned int trackCandidateIdx = modulesInGPU.trackCandidateModuleIndices[innerInnerInnerLowerModuleArrayIndex] + trackCandidateModuleIdx;
		                addTrackCandidateToMemory(trackCandidatesInGPU, trackCandidateType, innerObjectIndex, outerObjectIndex, trackCandidateIdx);
                    }

                }
            }

        }
    }

    //step 2 tracklet-triplet
    if(innerObjectArrayIndex < nTracklets)
    {
        innerObjectIndex = innerInnerInnerLowerModuleArrayIndex * N_MAX_TRACKLETS_PER_MODULE + innerObjectArrayIndex;
        unsigned int outerInnerInnerLowerModuleIndex = modulesInGPU.reverseLookupLowerModuleIndices[trackletsInGPU.lowerModuleIndices[4 * innerObjectIndex + 2]];//same as innerOuterInnerLowerModuleIndex
        if(outerObjectArrayIndex < fminf(tripletsInGPU.nTriplets[outerInnerInnerLowerModuleIndex],N_MAX_TRIPLETS_PER_MODULE))
        {
	        outerObjectIndex = outerInnerInnerLowerModuleIndex * N_MAX_TRIPLETS_PER_MODULE + outerObjectArrayIndex;
	        success = runTrackCandidateDefaultAlgoTrackletToTriplet(trackletsInGPU, tripletsInGPU, innerObjectIndex, outerObjectIndex,trackCandidateType);
	        if(success)
            {
	            unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[innerInnerInnerLowerModuleArrayIndex],1);
	            atomicAdd(&trackCandidatesInGPU.nTrackCandidatesT4T3[innerInnerInnerLowerModuleArrayIndex],1);
	            if(trackCandidateModuleIdx >= N_MAX_TRACK_CANDIDATES_PER_MODULE)
                {
                    #ifdef Warnings
		            if(trackCandidateModuleIdx == N_MAX_TRACK_CANDIDATES_PER_MODULE)
                    {
		                printf("Track Candidate excess alert! lower Module array index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                    }
                    #endif
                }
	            else
                {

		            if(modulesInGPU.trackCandidateModuleIndices[innerInnerInnerLowerModuleArrayIndex] == -1)
                    {
                        #ifdef Warnings
		                printf("Track candidates: no memory for module at module index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                        #endif
                    }
		            else
                    {
		                unsigned int trackCandidateIdx = modulesInGPU.trackCandidateModuleIndices[innerInnerInnerLowerModuleArrayIndex] + trackCandidateModuleIdx;

		                addTrackCandidateToMemory(trackCandidatesInGPU, trackCandidateType, innerObjectIndex, outerObjectIndex, trackCandidateIdx);
                    }
                }
            }
        }
    }
    //step 3 triplet-tracklet
    if(innerObjectArrayIndex < nTriplets)
    {
        innerObjectIndex = innerInnerInnerLowerModuleArrayIndex * N_MAX_TRIPLETS_PER_MODULE + innerObjectArrayIndex;
        unsigned int outerInnerInnerLowerModuleIndex = modulesInGPU.reverseLookupLowerModuleIndices[tripletsInGPU.lowerModuleIndices[3 * innerObjectIndex + 1]];//same as innerOuterInnerLowerModuleIndex

        if(outerObjectArrayIndex < fminf(trackletsInGPU.nTracklets[outerInnerInnerLowerModuleIndex],N_MAX_TRACKLETS_PER_MODULE))
        {
	        outerObjectIndex = outerInnerInnerLowerModuleIndex * N_MAX_TRACKLETS_PER_MODULE + outerObjectArrayIndex;
	        success = runTrackCandidateDefaultAlgoTripletToTracklet(trackletsInGPU, tripletsInGPU, innerObjectIndex, outerObjectIndex,trackCandidateType);
	        if(success)
            {
	            unsigned int trackCandidateModuleIdx = atomicAdd(&trackCandidatesInGPU.nTrackCandidates[innerInnerInnerLowerModuleArrayIndex],1);
	            atomicAdd(&trackCandidatesInGPU.nTrackCandidatesT3T4[innerInnerInnerLowerModuleArrayIndex],1);
	            if(trackCandidateModuleIdx >= N_MAX_TRACK_CANDIDATES_PER_MODULE)
                {
                    #ifdef Warnings
		            if(trackCandidateModuleIdx == N_MAX_TRACK_CANDIDATES_PER_MODULE)
		            printf("Track Candidate excess alert! Module index = %d\n",innerInnerInnerLowerModuleArrayIndex);
                   #endif
                }
	            else
                {
		            if(modulesInGPU.trackCandidateModuleIndices[innerInnerInnerLowerModuleArrayIndex] == -1)
                    {
                        #ifdef Warnings
		                printf("Track candidates: no memory for module at module index = %d, outer T4 module index = %d\n",innerInnerInnerLowerModuleArrayIndex, outerInnerInnerLowerModuleIndex);
                        #endif
                    }
		            else
                    {
		                unsigned int trackCandidateIdx = modulesInGPU.trackCandidateModuleIndices[innerInnerInnerLowerModuleArrayIndex] + trackCandidateModuleIdx;
		                addTrackCandidateToMemory(trackCandidatesInGPU, trackCandidateType, innerObjectIndex, outerObjectIndex, trackCandidateIdx);
                    }
                }
            }
        }
    }
}

__global__ void createPixelTripletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, unsigned int* connectedPixelSize, unsigned int* connectedPixelIndex, unsigned int nPixelSegments, unsigned int* seg_pix_gpu, unsigned int* seg_pix_gpu_offset, unsigned int totalSegs)
{
    //newgrid with map
    unsigned int offsetIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(offsetIndex >= totalSegs)  return;

    int segmentModuleIndex = seg_pix_gpu_offset[offsetIndex];
    int pixelSegmentArrayIndex = seg_pix_gpu[offsetIndex];
    if(pixelSegmentArrayIndex >= nPixelSegments) return;
    if(segmentModuleIndex >= connectedPixelSize[pixelSegmentArrayIndex]) return;

    unsigned int tripletLowerModuleIndex; //index of the module that connects to this pixel
    unsigned int tempIndex = connectedPixelIndex[pixelSegmentArrayIndex] + segmentModuleIndex; //gets module array index for segment

    //these are actual module indices
    tripletLowerModuleIndex = modulesInGPU.connectedPixels[tempIndex];
    unsigned int tripletLowerModuleArrayIndex = modulesInGPU.reverseLookupLowerModuleIndices[tripletLowerModuleIndex];
    if(tripletLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;

    unsigned int pixelModuleIndex = *modulesInGPU.nModules - 1;
    unsigned int nOuterTriplets = min(tripletsInGPU.nTriplets[tripletLowerModuleArrayIndex], N_MAX_TRIPLETS_PER_MODULE);

    if(nOuterTriplets == 0) return;
    if(modulesInGPU.moduleType[tripletLowerModuleIndex] == SDL::TwoS) return; //Removes 2S-2S

    //fetch the triplet
    unsigned int outerTripletArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if(outerTripletArrayIndex >= nOuterTriplets) return;
    unsigned int pixelSegmentIndex = pixelModuleIndex * N_MAX_SEGMENTS_PER_MODULE + pixelSegmentArrayIndex;
    unsigned int outerTripletIndex = tripletLowerModuleArrayIndex * N_MAX_TRIPLETS_PER_MODULE + outerTripletArrayIndex;
    if(modulesInGPU.moduleType[tripletsInGPU.lowerModuleIndices[3 * outerTripletIndex + 1]] == SDL::TwoS) return; //REMOVES PS-2S

    if(segmentsInGPU.isDup[pixelSegmentArrayIndex]) return;
    if(segmentsInGPU.partOfPT5[pixelSegmentArrayIndex]) return; //don't make pT3s for those pixels that are part of pT5
    if(tripletsInGPU.partOfPT5[outerTripletIndex]) return; //don't create pT3s for T3s accounted in pT5s

    float pixelRadius, pixelRadiusError, tripletRadius, rPhiChiSquared, rzChiSquared, rPhiChiSquaredInwards;
    bool success = runPixelTripletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, pixelSegmentIndex, outerTripletIndex, pixelRadius, pixelRadiusError, tripletRadius, rzChiSquared, rPhiChiSquared, rPhiChiSquaredInwards);

    if(success)
    {

        short layer2_adjustment;
        if(modulesInGPU.layers[tripletLowerModuleIndex] == 1){layer2_adjustment = 1;} //get upper segment to be in second layer
        else if( modulesInGPU.layers[tripletLowerModuleIndex] == 2){layer2_adjustment = 0;} // get lower segment to be in second layer
        else{return;} // ignore anything else
        float phi = hitsInGPU.phis[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTripletIndex]+layer2_adjustment]]];
        float eta = hitsInGPU.etas[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTripletIndex]+layer2_adjustment]]];
        float eta_pix = segmentsInGPU.eta[pixelSegmentArrayIndex];
        float phi_pix = segmentsInGPU.phi[pixelSegmentArrayIndex];
        float pt = segmentsInGPU.ptIn[pixelSegmentArrayIndex];
        float score = rPhiChiSquared+rPhiChiSquaredInwards;
        unsigned int pixelTripletIndex = atomicAdd(pixelTripletsInGPU.nPixelTriplets, 1);
        if(pixelTripletIndex >= N_MAX_PIXEL_TRIPLETS)
        {
            #ifdef Warnings
            if(pixelTripletIndex == N_MAX_PIXEL_TRIPLETS)
            {
               printf("Pixel Triplet excess alert!\n");
            }
            #endif
        }
        else
        {
#ifdef CUT_VALUE_DEBUG
            addPixelTripletToMemory(pixelTripletsInGPU, pixelSegmentIndex, outerTripletIndex, pixelRadius, pixelRadiusError, tripletRadius, rPhiChiSquared, rPhiChiSquaredInwards, rzChiSquared, pixelTripletIndex, pt, eta, phi, eta_pix, phi_pix, score);
#else
            addPixelTripletToMemory(pixelTripletsInGPU, pixelSegmentIndex, outerTripletIndex, pixelRadius,tripletRadius, pixelTripletIndex, pt,eta,phi,eta_pix,phi_pix,score);
#endif
        }
    }

}

__global__ void createQuintupletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, unsigned int* threadIdx_gpu, unsigned int* threadIdx_gpu_offset, int nTotalTriplets)
{
    int gidy = blockIdx.y * blockDim.y + threadIdx.y;
    int np = gridDim.y * blockDim.y;
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int iter=gidy; iter < nTotalTriplets; iter+=np)
    {
        //int lowerModuleArray1 = threadIdx_gpu[blockIdx.y * blockDim.y + threadIdx.y];
        int lowerModuleArray1 = threadIdx_gpu[iter];

        //this if statement never gets executed!
        if(lowerModuleArray1  >= *modulesInGPU.nLowerModules) continue;

        unsigned int nInnerTriplets = tripletsInGPU.nTriplets[lowerModuleArray1];

        //unsigned int innerTripletArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;
        //unsigned int innerTripletArrayIndex = threadIdx_gpu_offset[blockIdx.y * blockDim.y + threadIdx.y];
        //unsigned int outerTripletArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int innerTripletArrayIndex = threadIdx_gpu_offset[iter];
        unsigned int outerTripletArrayIndex = gidx;

        if(innerTripletArrayIndex >= nInnerTriplets) continue;

        unsigned int innerTripletIndex = lowerModuleArray1 * N_MAX_TRIPLETS_PER_MODULE + innerTripletArrayIndex;
        unsigned int lowerModule1 = modulesInGPU.lowerModuleIndices[lowerModuleArray1];
        //these are actual module indices!! not lower module indices!
        unsigned int lowerModule2 = tripletsInGPU.lowerModuleIndices[3 * innerTripletIndex + 1];
        unsigned int lowerModule3 = tripletsInGPU.lowerModuleIndices[3 * innerTripletIndex + 2];
        unsigned int lowerModuleArray3 = modulesInGPU.reverseLookupLowerModuleIndices[lowerModule3];
        unsigned int nOuterTriplets = min(tripletsInGPU.nTriplets[lowerModuleArray3], N_MAX_TRIPLETS_PER_MODULE);

        if(outerTripletArrayIndex >= nOuterTriplets) continue;
        unsigned int outerTripletIndex = lowerModuleArray3 * N_MAX_TRIPLETS_PER_MODULE + outerTripletArrayIndex;
        //these are actual module indices!!
        unsigned int lowerModule4 = tripletsInGPU.lowerModuleIndices[3 * outerTripletIndex + 1];
        unsigned int lowerModule5 = tripletsInGPU.lowerModuleIndices[3 * outerTripletIndex + 2];

        float innerRadius, innerRadiusMin, innerRadiusMin2S, innerRadiusMax, innerRadiusMax2S, outerRadius, outerRadiusMin, outerRadiusMin2S, outerRadiusMax, outerRadiusMax2S, bridgeRadius, bridgeRadiusMin, bridgeRadiusMin2S, bridgeRadiusMax, bridgeRadiusMax2S, regressionG, regressionF, regressionRadius, chiSquared, nonAnchorChiSquared; //required for making distributions

        bool success = runQuintupletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, lowerModule1, lowerModule2, lowerModule3, lowerModule4, lowerModule5, innerTripletIndex, outerTripletIndex, innerRadius, innerRadiusMin, innerRadiusMax, outerRadius, outerRadiusMin, outerRadiusMax, bridgeRadius, bridgeRadiusMin, bridgeRadiusMax, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S,
            outerRadiusMax2S, regressionG, regressionF, regressionRadius, chiSquared, nonAnchorChiSquared);

        if(success)
        {
            short layer2_adjustment;
            int layer = modulesInGPU.layers[lowerModule1];
            if(layer == 1)
            {
                layer2_adjustment = 1;
            } //get upper segment to be in second layer
            else if(layer == 2)
            {
                layer2_adjustment = 0;
            } // get lower segment to be in second layer
            else
            {
                return;
            } // ignore anything else TODO: move this to start, before object is made (faster)
            unsigned int quintupletModuleIndex = atomicAdd(&quintupletsInGPU.nQuintuplets[lowerModuleArray1], 1);
            if(quintupletModuleIndex >= N_MAX_QUINTUPLETS_PER_MODULE)
            {
#ifdef Warnings
                if(quintupletModuleIndex ==  N_MAX_QUINTUPLETS_PER_MODULE)
                {
                   printf("Quintuplet excess alert! Module index = %d\n", lowerModuleArray1);
                }
#endif
            }
            else
            {
                //this if statement should never get executed!
                if(modulesInGPU.quintupletModuleIndices[lowerModuleArray1] == -1)
                {
                    printf("Quintuplets : no memory for module at module index = %d\n", lowerModuleArray1);
                }
                else
                {
                    unsigned int quintupletIndex = modulesInGPU.quintupletModuleIndices[lowerModuleArray1] +  quintupletModuleIndex;
                    float phi = hitsInGPU.phis[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTripletIndex+layer2_adjustment]]]];
                    float eta = hitsInGPU.etas[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTripletIndex+layer2_adjustment]]]];
                    float pt = (innerRadius+outerRadius)*3.8*1.602/(2*100*5.39);
                //float scores[4];// still fills all values,but only rphi sum is actually used for the cuts. Others may be removed later.
                //scoreT5(modulesInGPU,hitsInGPU,mdsInGPU,segmentsInGPU,tripletsInGPU,innerTripletIndex,outerTripletIndex,layer,scores);
                //scores[0] = chiSquared;
                //scores[2] = chiSquared + nonAnchorChiSquared;
                    float scores = chiSquared + nonAnchorChiSquared;
#ifdef CUT_VALUE_DEBUG
                    addQuintupletToMemory(quintupletsInGPU, innerTripletIndex, outerTripletIndex, lowerModule1, lowerModule2, lowerModule3, lowerModule4, lowerModule5, innerRadius, innerRadiusMin, innerRadiusMax, outerRadius, outerRadiusMin, outerRadiusMax, bridgeRadius, bridgeRadiusMin, bridgeRadiusMax, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, regressionG, regressionF, regressionRadius, chiSquared, nonAnchorChiSquared,
                        pt, eta, phi, scores, layer, quintupletIndex);
#else
                    addQuintupletToMemory(quintupletsInGPU, innerTripletIndex, outerTripletIndex, lowerModule1, lowerModule2, lowerModule3, lowerModule4, lowerModule5, innerRadius, outerRadius, regressionG, regressionF, regressionRadius, pt,eta,phi,scores,layer,quintupletIndex);
#endif
                }
            }
        }
    }
}

__global__ void createPixelQuintupletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int* connectedPixelSize, unsigned int* connectedPixelIndex, unsigned int nPixelSegments, unsigned int* seg_pix_gpu, unsigned int* seg_pix_gpu_offset, unsigned int totalSegs)
{
    unsigned int offsetIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(offsetIndex >= totalSegs) return;

    int segmentModuleIndex = seg_pix_gpu_offset[offsetIndex];
    int pixelSegmentArrayIndex = seg_pix_gpu[offsetIndex];
    if(pixelSegmentArrayIndex >= nPixelSegments) return;
    if(segmentModuleIndex >= connectedPixelSize[pixelSegmentArrayIndex]) return;

    unsigned int quintupletLowerModuleIndex; //index of the module that connects to this pixel
    unsigned int tempIndex = connectedPixelIndex[pixelSegmentArrayIndex] + segmentModuleIndex; //gets module array index for segment

    //these are actual module indices
    quintupletLowerModuleIndex = modulesInGPU.connectedPixels[tempIndex];
    unsigned int quintupletLowerModuleArrayIndex = modulesInGPU.reverseLookupLowerModuleIndices[quintupletLowerModuleIndex];
    if(quintupletLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) return;

    unsigned int pixelModuleIndex = *modulesInGPU.nModules - 1;
    unsigned int nOuterQuintuplets = min(quintupletsInGPU.nQuintuplets[quintupletLowerModuleArrayIndex], N_MAX_QUINTUPLETS_PER_MODULE);

    if(nOuterQuintuplets == 0) return;

    //fetch the quintuplet
    unsigned int outerQuintupletArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if(outerQuintupletArrayIndex >= nOuterQuintuplets) return;
    unsigned int pixelSegmentIndex = pixelModuleIndex * N_MAX_SEGMENTS_PER_MODULE + pixelSegmentArrayIndex;

    unsigned int quintupletIndex = modulesInGPU.quintupletModuleIndices[quintupletLowerModuleArrayIndex] + outerQuintupletArrayIndex;

    if(segmentsInGPU.isDup[pixelSegmentArrayIndex]) return;//skip duplicated pLS
    if(quintupletsInGPU.isDup[quintupletIndex]) return; //skip duplicated T5s

    float rzChiSquared, rPhiChiSquared, rPhiChiSquaredInwards;

    bool success = runPixelQuintupletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, quintupletsInGPU, pixelSegmentIndex, quintupletIndex, rzChiSquared, rPhiChiSquared, rPhiChiSquaredInwards);

    if(success)
    {
        unsigned int pixelQuintupletIndex = atomicAdd(pixelQuintupletsInGPU.nPixelQuintuplets, 1);
        if(pixelQuintupletIndex >= N_MAX_PIXEL_QUINTUPLETS)
        {
            #ifdef Warnings
            if(pixelQuintupletIndex == N_MAX_PIXEL_QUINTUPLETS)
            {
               printf("Pixel Quintuplet excess alert!\n");
            }
            #endif
        }
        else
        {
#ifdef CUT_VALUE_DEBUG
            addPixelQuintupletToMemory(pixelQuintupletsInGPU, pixelSegmentIndex, quintupletIndex, pixelQuintupletIndex,rzChiSquared, rPhiChiSquared, rPhiChiSquaredInwards, rPhiChiSquared);

#else
            float eta = quintupletsInGPU.eta[quintupletIndex];
            float phi = quintupletsInGPU.phi[quintupletIndex];
            addPixelQuintupletToMemory(pixelQuintupletsInGPU, pixelSegmentIndex, quintupletIndex, pixelQuintupletIndex,rPhiChiSquaredInwards+rPhiChiSquared, eta,phi);
#endif
        }
    }
}



__device__ void scoreT5(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU,struct SDL::segments& segmentsInGPU,struct SDL::triplets& tripletsInGPU, unsigned int innerTrip, unsigned int outerTrip, int layer, float* scores)
{
    int hits1[10] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    hits1[0] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTrip]]]; // inner triplet inner segment inner md inner hit
    hits1[1] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTrip]]+1]; // inner triplet inner segment inner md outer hit
    hits1[2] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTrip]+1]]; // inner triplet inner segment outer md inner hit
    hits1[3] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTrip]+1]+1]; // inner triplet inner segment outer md outer hit
    hits1[4] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTrip+1]+1]]; // inner triplet outer segment outer md inner hit
    hits1[5] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTrip+1]+1]+1]; // inner triplet outer segment outer md outer hit
    hits1[6] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip+1]]]; // outer triplet outersegment inner md inner hit
    hits1[7] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip+1]]+1]; // outer triplet outersegment inner md outer hit
    hits1[8] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip+1]+1]]; // outer triplet outersegment outer md inner hit
    hits1[9] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip+1]+1]+1]; // outer triplet outersegment outer md outer hit

    unsigned int mod1 = hitsInGPU.moduleIndices[hits1[0]];
    SDL::ModuleLayerType type1 = modulesInGPU.moduleLayerType[mod1];
    unsigned int mod2 = hitsInGPU.moduleIndices[hits1[6-2*layer]];//4 for layer=1 (second hit in 3rd layer), 2 for layer=2 (second hit in third layer)
    SDL::ModuleLayerType type2 = modulesInGPU.moduleLayerType[mod1];
    float r1,r2,z1,z2;
    if(type1 == 0)
    {
        //lower hit is pixel
        r1 = hitsInGPU.rts[hits1[0]];
        z1 = hitsInGPU.zs[hits1[0]];
    }
    else
    {
        //upper hit is pixel
        r1 = hitsInGPU.rts[hits1[1]];
        z1 = hitsInGPU.zs[hits1[1]];
    }
    if(type2==0)
    {
        //lower hit is pixel
        r2 = hitsInGPU.rts[hits1[6-2*layer]];
        z2 = hitsInGPU.zs[hits1[6-2*layer]];
    }
    else
    {
        r2 = hitsInGPU.rts[hits1[7-2*layer]];
        z2 = hitsInGPU.zs[hits1[7-2*layer]];
    }
    float slope_barrel = (z2-z1)/(r2-r1);
    float slope_endcap = (r2-r1)/(z2-z1);

    //least squares
    float rsum=0, zsum=0, r2sum=0,rzsum=0;
    float rsum_e=0, zsum_e=0, r2sum_e=0,rzsum_e=0;
    for(int i =0; i < 10; i++)
    {
        rsum += hitsInGPU.rts[hits1[i]];
        zsum += hitsInGPU.zs[hits1[i]];
        r2sum += hitsInGPU.rts[hits1[i]]*hitsInGPU.rts[hits1[i]];
        rzsum += hitsInGPU.rts[hits1[i]]*hitsInGPU.zs[hits1[i]];

        rsum_e += hitsInGPU.zs[hits1[i]];
        zsum_e += hitsInGPU.rts[hits1[i]];
        r2sum_e += hitsInGPU.zs[hits1[i]]*hitsInGPU.zs[hits1[i]];
        rzsum_e += hitsInGPU.zs[hits1[i]]*hitsInGPU.rts[hits1[i]];
    }
    float slope_lsq = (10*rzsum - rsum*zsum)/(10*r2sum-rsum*rsum);
    float b = (r2sum*zsum-rsum*rzsum)/(r2sum*10-rsum*rsum);
    float slope_lsq_e = (10*rzsum_e - rsum_e*zsum_e)/(10*r2sum_e-rsum_e*rsum_e);
    float b_e = (r2sum_e*zsum_e-rsum_e*rzsum_e)/(r2sum_e*10-rsum_e*rsum_e);

    float score=0;
    float score_lsq=0;
    for( int i=0; i <10; i++)
    {
        float z = hitsInGPU.zs[hits1[i]];
        float r = hitsInGPU.rts[hits1[i]]; // cm
        float subdet = modulesInGPU.subdets[hitsInGPU.moduleIndices[hits1[i]]];
        float drdz = modulesInGPU.drdzs[hitsInGPU.moduleIndices[hits1[i]]];
        float var=0;
        float var_lsq=0;
        if(subdet == 5)
        {
            // 5== barrel
            var = slope_barrel*(r-r1) - (z-z1);
            var_lsq = slope_lsq*(r-r1) - (z-z1);
        }
        else
        {
            var = slope_endcap*(z-z1) - (r-r1);
            var_lsq = slope_lsq_e*(z-z1) - (r-r1);
        }
        float err;
        if(modulesInGPU.moduleLayerType[hitsInGPU.moduleIndices[hits1[i]]]==0)
        {
            err=0.15*cos(atan(drdz));//(1.5mm)^2
        }
        else
        {
            err=5.0*cos(atan(drdz));
        }//(5cm)^2
        score += (var*var) / (err*err);
        score_lsq += (var_lsq*var_lsq) / (err*err);
    }
    //printf("%f %f\n",score,score_lsq);
    scores[1] = score;
    scores[3] = score_lsq;
    //return score;
}
__device__ int inline checkHitsT5(unsigned int ix, unsigned int jx,struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU,struct SDL::quintuplets& quintupletsInGPU)
{
    unsigned int hits1[10];// = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    unsigned int hits2[10];// = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    hits1[0] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*ix]]]]; // inner triplet inner segment inner md inner hit
    hits1[1] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*ix]]]+1]; // inner triplet inner segment inner md outer hit
    hits1[2] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*ix]]+1]]; // inner triplet inner segment outer md inner hit
    hits1[3] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*ix]]+1]+1]; // inner triplet inner segment outer md outer hit
    hits1[4] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*ix]+1]+1]]; // inner triplet outer segment outer md inner hit
    hits1[5] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*ix]+1]+1]+1]; // inner triplet outer segment outer md outer hit
    hits1[6] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*ix+1]+1]]]; // outer triplet outersegment inner md inner hit
    hits1[7] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*ix+1]+1]]+1]; // outer triplet outersegment inner md outer hit
    hits1[8] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*ix+1]+1]+1]]; // outer triplet outersegment outer md inner hit
    hits1[9] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*ix+1]+1]+1]+1]; // outer triplet outersegment outer md outer hit

    hits2[0] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*jx]]]]; // inner triplet inner segment inner md inner hit
    hits2[1] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*jx]]]+1]; // inner triplet inner segment inner md outer hit
    hits2[2] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*jx]]+1]]; // inner triplet inner segment outer md inner hit
    hits2[3] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*jx]]+1]+1]; // inner triplet inner segment outer md outer hit
    hits2[4] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*jx]+1]+1]]; // inner triplet outer segment outer md inner hit
    hits2[5] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*jx]+1]+1]+1]; // inner triplet outer segment outer md outer hit
    hits2[6] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*jx+1]+1]]]; // outer triplet outersegment inner md inner hit
    hits2[7] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*jx+1]+1]]+1]; // outer triplet outersegment inner md outer hit
    hits2[8] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*jx+1]+1]+1]]; // outer triplet outersegment outer md inner hit
    hits2[9] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*jx+1]+1]+1]+1]; // outer triplet outersegment outer md outer hit

    int nMatched =0;
    for (int i =0; i<10;i++)
    {
        bool matched = false;
        for (int j =0; j<10; j++)
        {
            if(hits1[i] == hits2[j])
            {
                matched = true; break;
            }
        }
        if(matched){nMatched++;}
    }
    return nMatched;
}

__device__ int duplicateCounter;
__global__ void removeDupQuintupletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU,bool secondPass)
{
    int dup_count=0;
    int nLowerModules = *modulesInGPU.nLowerModules;
    int blockxSize = blockDim.x*gridDim.x;
    int blockySize = blockDim.y*gridDim.y;
    for(unsigned int lowmod1=blockIdx.x*blockDim.x+threadIdx.x; lowmod1<nLowerModules;lowmod1+=blockxSize)
    {
        int nQuintuplets_lowmod1 = quintupletsInGPU.nQuintuplets[lowmod1];
        int quintupletModuleIndices_lowmod1 = modulesInGPU.quintupletModuleIndices[lowmod1];
        for(unsigned int ix1=blockIdx.y*blockDim.y+threadIdx.y; ix1<nQuintuplets_lowmod1; ix1+=blockySize)
        {
            unsigned int ix = quintupletModuleIndices_lowmod1 + ix1;
            if(secondPass && (quintupletsInGPU.partOfPT5[ix] || quintupletsInGPU.isDup[ix]))
            {
                continue;
            }
            float pt1  = quintupletsInGPU.pt[ix];
            float eta1 = quintupletsInGPU.eta[ix];
            float phi1 = quintupletsInGPU.phi[ix];
            bool isDup = false;
	    float score_rphisum1 = quintupletsInGPU.score_rphisum[ix];
            for(unsigned int lowmod=0; lowmod<nLowerModules;lowmod++)
            {
	        int nQuintuplets_lowmod = quintupletsInGPU.nQuintuplets[lowmod];
                int quintupletModuleIndices_lowmod = modulesInGPU.quintupletModuleIndices[lowmod];
                for(unsigned int jx1=0; jx1<nQuintuplets_lowmod; jx1++)
                {
                    unsigned int jx = quintupletModuleIndices_lowmod + jx1;
                    if(ix==jx)
                    {
                        continue;
                    }
                    if(secondPass && (quintupletsInGPU.partOfPT5[jx] || quintupletsInGPU.isDup[jx]))
                    {
                        continue;
                    }
                    float pt2  = quintupletsInGPU.pt[jx];
                    float eta2 = quintupletsInGPU.eta[jx];
                    float phi2 = quintupletsInGPU.phi[jx];
                    float dEta = fabsf(eta1-eta2);
                    float dPhi = fabsf(phi1-phi2);
		    float score_rphisum2 = quintupletsInGPU.score_rphisum[jx];
                    if (dEta > 0.1)
                    {
                        continue;
                    }
                    if(dPhi > M_PI){dPhi = dPhi - 2*M_PI;}
                    if (abs(dPhi) > 0.1){continue;}
                    float dR2 = dEta*dEta + dPhi*dPhi;
                    int nMatched = checkHitsT5(ix,jx,mdsInGPU,segmentsInGPU,tripletsInGPU,quintupletsInGPU);
                    if(secondPass && (dR2 < 0.001 || nMatched >= 5))
                    {
                        if(score_rphisum1 > score_rphisum2 )
                        {
                            rmQuintupletToMemory(quintupletsInGPU,ix);
                            continue;
                        }
                        if( (score_rphisum1 == score_rphisum2) && (ix<jx))
                        {
                            rmQuintupletToMemory(quintupletsInGPU,ix);
                            continue;
                         }
                    }
                    if(nMatched >=7)
                    {
                        dup_count++;
                        if( score_rphisum1 > score_rphisum2 )
                        {
                            rmQuintupletToMemory(quintupletsInGPU,ix);continue;
                        }
                        if( (score_rphisum1 == score_rphisum2) && (ix<jx))
                        {
                            rmQuintupletToMemory(quintupletsInGPU,ix);continue;
                        }
                    }
                }
            }
        }
    }
}

__device__ float scorepT3(struct SDL::modules& modulesInGPU,struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU,struct SDL::segments& segmentsInGPU,struct SDL::triplets& tripletsInGPU, unsigned int innerPix, unsigned int outerTrip, float pt, float pz)
{
    unsigned int hits1[10];// = {-1,-1,-1,-1};
    hits1[0] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*innerPix]];
    hits1[1] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*innerPix]+1];
    hits1[2] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*innerPix+1]];
    hits1[3] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*innerPix+1]+1];
    hits1[4] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip]]];// outer trip, inner seg, inner md, inner hit
    hits1[5] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip]]+1];// o t, is, im oh
    hits1[6] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip]+1]]; //ot is om ih
    hits1[7] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip]+1]+1];
    hits1[8] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip+1]+1]];// ot os om ih
    hits1[9] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip+1]+1]+1];

    float r1 = hitsInGPU.rts[hits1[0]];
    float z1 = hitsInGPU.zs[hits1[0]];
    float r2 = hitsInGPU.rts[hits1[3]];
    float z2 = hitsInGPU.zs[hits1[3]];

    float slope_barrel = (z2-z1)/(r2-r1);
    float slope_endcap = (r2-r1)/(z2-z1);

    float score = 0;
    for(unsigned int i=4; i <10; i++)
    {
        float z = hitsInGPU.zs[hits1[i]];
        float r = hitsInGPU.rts[hits1[i]]; // cm
        float subdet = modulesInGPU.subdets[hitsInGPU.moduleIndices[hits1[i]]];
        float drdz = modulesInGPU.drdzs[hitsInGPU.moduleIndices[hits1[i]]];
        float var=0;
        if(subdet == 5)
        {// 5== barrel
            var = slope_barrel*(r-r1) - (z-z1);
        }
        else
        {
            var = slope_endcap*(z-z1) - (r-r1);
        }
        float err;
        if(modulesInGPU.moduleLayerType[hitsInGPU.moduleIndices[hits1[i]]]==0)
        {
            err=0.15*cos(atan(drdz));//(1.5mm)^2
        }
        else
        {
            err=5.0*cos(atan(drdz));
        }//(5cm)^2
        score += (var*var) / (err*err);
    }
    //printf("pT3 score: %f\n",score);
    return score;
}
__device__ inline int* checkHitspT3(unsigned int ix, unsigned int jx,struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU,struct SDL::hits& hitsInGPU)
{
    int phits1[4] = {-1,-1,-1,-1};
    int phits2[4] = {-1,-1,-1,-1};
    phits1[0] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*pixelTripletsInGPU.pixelSegmentIndices[ix]]]];
    phits1[1] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*pixelTripletsInGPU.pixelSegmentIndices[ix]+1]]];
    phits1[2] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*pixelTripletsInGPU.pixelSegmentIndices[ix]]+1]];
    phits1[3] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*pixelTripletsInGPU.pixelSegmentIndices[ix]+1]+1]];

    phits2[0] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*pixelTripletsInGPU.pixelSegmentIndices[jx]]]];
    phits2[1] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*pixelTripletsInGPU.pixelSegmentIndices[jx]+1]]];
    phits2[2] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*pixelTripletsInGPU.pixelSegmentIndices[jx]]+1]];
    phits2[3] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*pixelTripletsInGPU.pixelSegmentIndices[jx]+1]+1]];

    int npMatched =0;
    for (int i =0; i<4;i++)
    {
        bool pmatched = false;
        for (int j =0; j<4; j++)
        {
            if(phits1[i] == phits2[j])
            {
                pmatched = true;
                break;
            }
        }
        if(pmatched)
        {
            npMatched++;
        }
    }
    int hits1[6] = {-1,-1,-1,-1,-1,-1};
    int hits2[6] = {-1,-1,-1,-1,-1,-1};
    hits1[0] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[ix]]]];// outer trip, inner seg, inner md, inner hit
    hits1[1] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[ix]]]+1];// o t, is, im oh
    hits1[2] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[ix]]+1]]; //ot is om ih
    hits1[3] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[ix]]+1]+1];
    hits1[4] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[ix]+1]+1]];// ot os om ih
    hits1[5] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[ix]+1]+1]+1];

    hits2[0] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[jx]]]];// outer trip, inner seg, inner md, inner hit
    hits2[1] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[jx]]]+1];// o t, is, im oh
    hits2[2] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[jx]]+1]]; //ot is om ih
    hits2[3] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[jx]]+1]+1];
    hits2[4] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[jx]+1]+1]];// ot os om ih
    hits2[5] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[jx]+1]+1]+1];

    int nMatched =0;
    for (int i =0; i<6;i++)
    {
        bool matched = false;
        for (int j =0; j<6; j++)
        {
            if(hits1[i] == hits2[j])
            {
                matched = true;
                break;
            }
        }
        if(matched){nMatched++;}
    }

    //if((nMatched >= 6) & (npMatched >= 4)){return true;}
    //if((nMatched >= 2) & (npMatched >= 1)){return true;}
     //if((nMatched  + npMatched >= 10)){return true;}
    int matched[2] = {npMatched, nMatched};
    return matched;//nMatched+npMatched;
}

__device__ int duplicateCounter_pT3 =0;

__global__ void removeDupPixelTripletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::triplets& tripletsInGPU, bool secondPass)
{
    int dup_count=0;
    int nPixelTriplets = *pixelTripletsInGPU.nPixelTriplets;
    int blockxSize = blockDim.x*gridDim.x;
    for (unsigned int ix=blockIdx.x*blockDim.x+threadIdx.x; ix<nPixelTriplets; ix+=blockxSize)
    {
        bool isDup = false;
	float score1 = pixelTripletsInGPU.score[ix];
        for (unsigned int jx=0; jx<nPixelTriplets; jx++)
        {
	    float score2 = pixelTripletsInGPU.score[jx];
            if(ix==jx)
            {
                continue;
            }
            int* nMatched = checkHitspT3(ix,jx,mdsInGPU,segmentsInGPU,tripletsInGPU,pixelTripletsInGPU,hitsInGPU);
            if(((nMatched[0] + nMatched[1]) >= 5) )
            {
                dup_count++;
                if( score1 > score2)
                {
                    rmPixelTripletToMemory(pixelTripletsInGPU,ix);
                    break;
                }
                if( (score1 == score2) && (ix<jx))
                {
                    rmPixelTripletToMemory(pixelTripletsInGPU,ix);
                    break;
                }
            }
        }
    }
}

__global__ void markUsedObjects(struct SDL::modules& modulesInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::quintuplets& quintupletsInGPU)
{
    for (unsigned int ix=blockIdx.x*blockDim.x+threadIdx.x; ix<*pixelQuintupletsInGPU.nPixelQuintuplets; ix+=blockDim.x*gridDim.x)
    {
        //mark the relevant T5 and pT3 here!
        if(pixelQuintupletsInGPU.isDup[ix])
        {
            continue;
        }
        unsigned int quintupletIndex = pixelQuintupletsInGPU.T5Indices[ix];
        unsigned int pixelSegmentArrayIndex = pixelQuintupletsInGPU.pixelIndices[ix]- ((*modulesInGPU.nModules - 1)* N_MAX_SEGMENTS_PER_MODULE);
        quintupletsInGPU.partOfPT5[quintupletIndex] = true;
        unsigned int innerTripletIndex = quintupletsInGPU.tripletIndices[2 * quintupletIndex];
        unsigned int outerTripletIndex = quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1];
        tripletsInGPU.partOfPT5[innerTripletIndex] = true;
        tripletsInGPU.partOfPT5[outerTripletIndex] = true;
        segmentsInGPU.partOfPT5[pixelSegmentArrayIndex] = true;
    }
}

__global__ void removeDupPixelQuintupletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::quintuplets& quintupletsInGPU, bool secondPass)
{
    //printf("running pT5 duprm\n");
    int dup_count=0;
    int nPixelQuintuplets = *pixelQuintupletsInGPU.nPixelQuintuplets;
    int blockxSize=blockDim.x*gridDim.x;
    for (unsigned int ix=blockIdx.x*blockDim.x+threadIdx.x; ix<nPixelQuintuplets; ix+=blockxSize)
    {
        bool isDup = false;
        if(secondPass && pixelQuintupletsInGPU.isDup[ix])
        {
            continue;
        }
	float score1 = pixelQuintupletsInGPU.score[ix];
        for (unsigned int jx=0; jx<nPixelQuintuplets; jx++)
        {
            if(ix==jx)
            {
                continue;
            }
            if(secondPass && pixelQuintupletsInGPU.isDup[jx])
            {
                continue;
            }
            unsigned int T5_ix = pixelQuintupletsInGPU.T5Indices[ix];
            unsigned int T5_jx = pixelQuintupletsInGPU.T5Indices[jx];
            unsigned int pLS_ix = pixelQuintupletsInGPU.pixelIndices[ix];
            unsigned int pLS_jx = pixelQuintupletsInGPU.pixelIndices[jx];
            int nMatched = checkHitsT5(T5_ix,T5_jx,mdsInGPU,segmentsInGPU,tripletsInGPU,quintupletsInGPU);
            int npMatched = checkHitspT5(pLS_ix,pLS_jx,mdsInGPU,segmentsInGPU,hitsInGPU);
	    float score2 = pixelQuintupletsInGPU.score[jx];
            if(((nMatched + npMatched) >=7))// || (secondPass && ((nMatched + npMatched) >=1)))
            {
                dup_count++;
                if( score1 > score2)
                {
                    rmPixelQuintupletToMemory(pixelQuintupletsInGPU,ix);
                    break;
                }
                if( (score1 == score2) && (ix>jx))
                {
                    rmPixelQuintupletToMemory(pixelQuintupletsInGPU,ix);
                    break;
                }
            }
        }
    }
}

__global__ void checkHitspLS(struct SDL::modules& modulesInGPU,struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::hits& hitsInGPU,bool secondpass)
{
    int counter=0;
    int pixelModuleIndex = *modulesInGPU.nModules - 1;
    unsigned int prefix = pixelModuleIndex*N_MAX_SEGMENTS_PER_MODULE;
    unsigned int nPixelSegments = segmentsInGPU.nSegments[pixelModuleIndex];
    if(nPixelSegments >  N_MAX_PIXEL_SEGMENTS_PER_MODULE)
    {
        nPixelSegments =  N_MAX_PIXEL_SEGMENTS_PER_MODULE;
    }
    for(int ix=blockIdx.x*blockDim.x+threadIdx.x;ix<nPixelSegments;ix+=blockDim.x*gridDim.x)
    {
        if(secondpass && (!segmentsInGPU.isQuad[ix] || segmentsInGPU.isDup[ix])){continue;}
        bool found=false;
        unsigned int phits1[4] ;
        //bool isQuad_ix = segmentsInGPU.isQuad[ix];
        //printf("isQuad %d\n",isQuad_ix);
        phits1[0] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*(prefix+ix)]]];
        phits1[1] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*(prefix+ix)+1]]];
        phits1[2] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*(prefix+ix)]+1]];
        phits1[3] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*(prefix+ix)+1]+1]];
        float eta_pix1 = segmentsInGPU.eta[ix];
        float phi_pix1 = segmentsInGPU.phi[ix];
        float pt1 = segmentsInGPU.ptIn[ix];
        for(int jx=0;jx<nPixelSegments;jx++)
        {
            if(secondpass && (!segmentsInGPU.isQuad[jx] || segmentsInGPU.isDup[jx])){continue;}
            if(ix==jx)
            {
                continue;
            }
            //bool isQuad_jx = segmentsInGPU.isQuad[jx];
            int quad_diff = segmentsInGPU.isQuad[ix] -segmentsInGPU.isQuad[jx];
            float ptErr_diff = segmentsInGPU.ptIn[ix] -segmentsInGPU.ptIn[jx];
            float score_diff = segmentsInGPU.score[ix] -segmentsInGPU.score[jx];
            if( (quad_diff > 0 )|| (score_diff<0 && quad_diff ==0))
            //if( (quad_diff > 0 )|| (ptErr_diff>0 && quad_diff ==0))
            {
                continue;
            }// always keep quads over trips. If they are the same, we want the object with the lower pt Error

            unsigned int phits2[4] ;
            phits2[0] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*(prefix+jx)]]];
            phits2[1] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*(prefix+jx)+1]]];
            phits2[2] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*(prefix+jx)]+1]];
            phits2[3] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*(prefix+jx)+1]+1]];
            float eta_pix2 = segmentsInGPU.eta[jx];
            float phi_pix2 = segmentsInGPU.phi[jx];
            float pt2 = segmentsInGPU.ptIn[jx];
            //if(abs(1/pt1 - 1/pt2)> 0.1)
            //{
            //    continue;
            //}
            int npMatched =0;
            for (int i =0; i<4;i++)
            {
                bool pmatched = false;
                for (int j =0; j<4; j++)
                {
                    if(phits1[i] == phits2[j])
                    {
                        pmatched = true;
                        break;
                    }
                }
                if(pmatched)
                {
                    npMatched++;
                }
            }
            if((npMatched ==4) && (ix < jx))
            { // if exact match, remove only 1
                found=true;
                break;
            }
            if(npMatched ==3)
            //if(npMatched >=2)
            {
                found=true;
                break;
            }
            if(secondpass){
              //printf("secondpass\n");
              if(npMatched >=1)
              {
                  found=true;
                  break;
              }
              float dEta = abs(eta_pix1-eta_pix2);
              float dPhi = abs(phi_pix1-phi_pix2);
              if(dPhi > M_PI){dPhi = dPhi - 2*M_PI;}
              //if(abs(dPhi) > 0.03){continue;}
              //if(abs(1./pt1 - 1./pt2) > 0.5){continue;}
              float dR2 = dEta*dEta + dPhi*dPhi;
              //if(dR2 <0.0003)
              if(dR2 <0.00075)
              {
                  found=true;
                  break;
              }
            }
        }
        if(found){counter++;rmPixelSegmentFromMemory(segmentsInGPU,ix);continue;}
    }
}
