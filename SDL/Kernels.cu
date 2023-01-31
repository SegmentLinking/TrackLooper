# include "Kernels.cuh"
# include "Constants.h"
# include "Constants.cuh"
# include "Hit.cuh"


typedef struct
{
    unsigned int index;
    short layer, subdet, side, moduleType, moduleLayerType, ring, rod;
    float sdMuls, drdz, moduleGapSize, miniTilt;
    bool isTilted;
    int moduleIndex, nConnectedModules;
}sharedModule;

__device__ void importModuleInfo(struct SDL::modules& modulesInGPU, sharedModule& module, int moduleArrayIndex)
{
    module.index = moduleArrayIndex;
    module.nConnectedModules = modulesInGPU.nConnectedModules[moduleArrayIndex];
    module.layer = modulesInGPU.layers[moduleArrayIndex];
    module.ring = modulesInGPU.rings[moduleArrayIndex];
    module.subdet = modulesInGPU.subdets[moduleArrayIndex];
    module.rod = modulesInGPU.rods[moduleArrayIndex];
    module.side = modulesInGPU.sides[moduleArrayIndex];
    module.moduleType = modulesInGPU.moduleType[moduleArrayIndex];
    module.moduleLayerType = modulesInGPU.moduleLayerType[moduleArrayIndex];
    module.isTilted = modulesInGPU.subdets[moduleArrayIndex] == SDL::Barrel and modulesInGPU.sides[moduleArrayIndex] != SDL::Center;
    module.drdz = module.moduleLayerType == SDL::Strip ? modulesInGPU.drdzs[moduleArrayIndex] : modulesInGPU.drdzs[modulesInGPU.partnerModuleIndices[moduleArrayIndex]];
    module.moduleGapSize = SDL::moduleGapSize_seg(module.layer, module.ring, module.subdet, module.side, module.rod);
    module.miniTilt =  module.isTilted ? (0.5f * SDL::pixelPSZpitch * module.drdz / sqrtf(1.f + module.drdz * module.drdz) / module.moduleGapSize) : 0;
 
}

__device__ void rmQuintupletToMemory(struct SDL::quintuplets& quintupletsInGPU,unsigned int quintupletIndex)
{
    quintupletsInGPU.isDup[quintupletIndex] = true;

}
__device__ void rmPixelTripletToMemory(struct SDL::pixelTriplets& pixelTripletsInGPU,unsigned int pixelTripletIndex)
{
    pixelTripletsInGPU.isDup[pixelTripletIndex] = 1;
}
__device__ void rmPixelQuintupletToMemory(struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int pixelQuintupletIndex)
{

    pixelQuintupletsInGPU.isDup[pixelQuintupletIndex] = 1;
}
__device__ void rmPixelSegmentFromMemory(struct SDL::segments& segmentsInGPU,unsigned int pixelSegmentArrayIndex){
    segmentsInGPU.isDup[pixelSegmentArrayIndex] = 1;
}


__device__ void scoreT5(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU,struct SDL::segments& segmentsInGPU,struct SDL::triplets& tripletsInGPU, unsigned int innerTrip, unsigned int outerTrip, int layer, float* scores)
{
    int hits1[10] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    hits1[0] = mdsInGPU.anchorHitIndices[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTrip]]]; // inner triplet inner segment inner md inner hit
    hits1[1] = mdsInGPU.outerHitIndices[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTrip]]]; // inner triplet inner segment inner md outer hit
    hits1[2] = mdsInGPU.anchorHitIndices[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTrip]+1]]; // inner triplet inner segment outer md inner hit
    hits1[3] = mdsInGPU.outerHitIndices[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTrip]+1]]; // inner triplet inner segment outer md outer hit
    hits1[4] = mdsInGPU.anchorHitIndices[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTrip+1]+1]]; // inner triplet outer segment outer md inner hit
    hits1[5] = mdsInGPU.outerHitIndices[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTrip+1]+1]]; // inner triplet outer segment outer md outer hit
    hits1[6] = mdsInGPU.anchorHitIndices[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip+1]]]; // outer triplet outersegment inner md inner hit
    hits1[7] = mdsInGPU.outerHitIndices[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip+1]]]; // outer triplet outersegment inner md outer hit
    hits1[8] = mdsInGPU.anchorHitIndices[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip+1]+1]]; // outer triplet outersegment outer md inner hit
    hits1[9] = mdsInGPU.outerHitIndices[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip+1]+1]]; // outer triplet outersegment outer md outer hit

    unsigned int mod1 = hitsInGPU.moduleIndices[hits1[0]];
    SDL::ModuleLayerType type1 = modulesInGPU.moduleLayerType[mod1];
    //unsigned int mod2 = hitsInGPU.moduleIndices[hits1[6-2*layer]];//4 for layer=1 (second hit in 3rd layer), 2 for layer=2 (second hit in third layer)
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
            err=0.15f*cosf(atanf(drdz));//(1.5mm)^2
        }
        else
        {
            err=5.0f*cosf(atanf(drdz));
        }//(5cm)^2
        score += (var*var) / (err*err);
        score_lsq += (var_lsq*var_lsq) / (err*err);
    }
    scores[1] = score;
    scores[3] = score_lsq;
}

__device__ int inline checkHitsT5(unsigned int ix, unsigned int jx,struct SDL::quintuplets& quintupletsInGPU)
{
    unsigned int hits1[10];// = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    unsigned int hits2[10];// = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};

    hits1[0] = quintupletsInGPU.hitIndices[10*ix];
    hits1[1] = quintupletsInGPU.hitIndices[10*ix+1];
    hits1[2] = quintupletsInGPU.hitIndices[10*ix+2];
    hits1[3] = quintupletsInGPU.hitIndices[10*ix+3];
    hits1[4] = quintupletsInGPU.hitIndices[10*ix+4];
    hits1[5] = quintupletsInGPU.hitIndices[10*ix+5];
    hits1[6] = quintupletsInGPU.hitIndices[10*ix+6];
    hits1[7] = quintupletsInGPU.hitIndices[10*ix+7];
    hits1[8] = quintupletsInGPU.hitIndices[10*ix+8];
    hits1[9] = quintupletsInGPU.hitIndices[10*ix+9];


    hits2[0] = quintupletsInGPU.hitIndices[10*jx];
    hits2[1] = quintupletsInGPU.hitIndices[10*jx+1];
    hits2[2] = quintupletsInGPU.hitIndices[10*jx+2];
    hits2[3] = quintupletsInGPU.hitIndices[10*jx+3];
    hits2[4] = quintupletsInGPU.hitIndices[10*jx+4];
    hits2[5] = quintupletsInGPU.hitIndices[10*jx+5];
    hits2[6] = quintupletsInGPU.hitIndices[10*jx+6];
    hits2[7] = quintupletsInGPU.hitIndices[10*jx+7];
    hits2[8] = quintupletsInGPU.hitIndices[10*jx+8];
    hits2[9] = quintupletsInGPU.hitIndices[10*jx+9];

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
__device__ int inline checkHitspT5(unsigned int ix, unsigned int jx,struct SDL::pixelQuintuplets& pixelQuintupletsInGPU)
{
    unsigned int hits1[14];// = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    unsigned int hits2[14];// = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
//    unsigned int* hits1 = &pixelQuintupletsInGPU.hitIndices[14*ix];
//    unsigned int* hits2 = &pixelQuintupletsInGPU.hitIndices[14*jx];

    hits1[0] = pixelQuintupletsInGPU.hitIndices[14*ix];
    hits1[1] = pixelQuintupletsInGPU.hitIndices[14*ix+1];
    hits1[2] = pixelQuintupletsInGPU.hitIndices[14*ix+2];
    hits1[3] = pixelQuintupletsInGPU.hitIndices[14*ix+3];
    hits1[4] = pixelQuintupletsInGPU.hitIndices[14*ix+4];
    hits1[5] = pixelQuintupletsInGPU.hitIndices[14*ix+5];
    hits1[6] = pixelQuintupletsInGPU.hitIndices[14*ix+6];
    hits1[7] = pixelQuintupletsInGPU.hitIndices[14*ix+7];
    hits1[8] = pixelQuintupletsInGPU.hitIndices[14*ix+8];
    hits1[9] = pixelQuintupletsInGPU.hitIndices[14*ix+9];
    hits1[10] = pixelQuintupletsInGPU.hitIndices[14*ix+10];
    hits1[11] = pixelQuintupletsInGPU.hitIndices[14*ix+11];
    hits1[12] = pixelQuintupletsInGPU.hitIndices[14*ix+12];
    hits1[13] = pixelQuintupletsInGPU.hitIndices[14*ix+13];


    hits2[0] = pixelQuintupletsInGPU.hitIndices[14*jx];
    hits2[1] = pixelQuintupletsInGPU.hitIndices[14*jx+1];
    hits2[2] = pixelQuintupletsInGPU.hitIndices[14*jx+2];
    hits2[3] = pixelQuintupletsInGPU.hitIndices[14*jx+3];
    hits2[4] = pixelQuintupletsInGPU.hitIndices[14*jx+4];
    hits2[5] = pixelQuintupletsInGPU.hitIndices[14*jx+5];
    hits2[6] = pixelQuintupletsInGPU.hitIndices[14*jx+6];
    hits2[7] = pixelQuintupletsInGPU.hitIndices[14*jx+7];
    hits2[8] = pixelQuintupletsInGPU.hitIndices[14*jx+8];
    hits2[9] = pixelQuintupletsInGPU.hitIndices[14*jx+9];
    hits2[10] = pixelQuintupletsInGPU.hitIndices[14*jx+10];
    hits2[11] = pixelQuintupletsInGPU.hitIndices[14*jx+11];
    hits2[12] = pixelQuintupletsInGPU.hitIndices[14*jx+12];
    hits2[13] = pixelQuintupletsInGPU.hitIndices[14*jx+13];

    int nMatched =0;
    for (int i =0; i<14;i++)
    {
        bool matched = false;
        for (int j =0; j<14; j++)
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

__global__ void removeDupQuintupletsInGPUAfterBuild(struct SDL::modules& modulesInGPU, struct SDL::quintuplets& quintupletsInGPU,struct SDL::objectRanges& rangesInGPU)
{
    int nLowerModules = *modulesInGPU.nLowerModules;
    int blockxSize = blockDim.x*gridDim.x;
    int blockySize = blockDim.y*gridDim.y;
    int blockzSize = blockDim.z*gridDim.z;

    for(unsigned int lowmod1=blockIdx.z*blockDim.z+threadIdx.z; lowmod1<nLowerModules;lowmod1+=blockzSize)
    {
        int nQuintuplets_lowmod1 = quintupletsInGPU.nQuintuplets[lowmod1];
        int quintupletModuleIndices_lowmod1 = rangesInGPU.quintupletModuleIndices[lowmod1];
        for(unsigned int ix1=blockIdx.y*blockDim.y+threadIdx.y; ix1<nQuintuplets_lowmod1; ix1+=blockySize)
        {
            unsigned int ix = quintupletModuleIndices_lowmod1 + ix1;
            float pt1  = __H2F(quintupletsInGPU.pt[ix]);
            float eta1 = __H2F(quintupletsInGPU.eta[ix]);
            float phi1 = __H2F(quintupletsInGPU.phi[ix]);
            float score_rphisum1 = __H2F(quintupletsInGPU.score_rphisum[ix]);
            int nQuintuplets_lowmod = quintupletsInGPU.nQuintuplets[lowmod1];
            int quintupletModuleIndices_lowmod = rangesInGPU.quintupletModuleIndices[lowmod1];

            for(unsigned int jx1=blockIdx.x*blockDim.x+threadIdx.x; jx1<nQuintuplets_lowmod; jx1+=blockxSize)
            {
                unsigned int jx = quintupletModuleIndices_lowmod + jx1;
                if(ix==jx){continue;}
                float pt2  = __H2F(quintupletsInGPU.pt[jx]);
                float eta2 = __H2F(quintupletsInGPU.eta[jx]);
                float phi2 = __H2F(quintupletsInGPU.phi[jx]);
                float dEta = fabsf(eta1-eta2);
                float dPhi = fabsf(phi1-phi2);
                float score_rphisum2 = __H2F(quintupletsInGPU.score_rphisum[jx]);

                if (dEta > 0.1f){continue;}
                if(dPhi > float(M_PI)){dPhi = dPhi - 2*float(M_PI);}
                if (abs(dPhi) > 0.1f){continue;}
                float dR2 = dEta*dEta + dPhi*dPhi;
                int nMatched = checkHitsT5(ix,jx,quintupletsInGPU);
                if(nMatched >=7)
                {
                    if( score_rphisum1 > score_rphisum2 )
                    {
                        rmQuintupletToMemory(quintupletsInGPU,ix);continue;
                    }
                    else if( (score_rphisum1 == score_rphisum2) && (ix<jx))
                    {
                        rmQuintupletToMemory(quintupletsInGPU,ix);continue;
                    }
                    else
                    {
                        rmQuintupletToMemory(quintupletsInGPU,jx);continue;
                    }
                }
            }
        }
    }
}
__global__ void removeDupQuintupletsInGPUBeforeTC(struct SDL::quintuplets& quintupletsInGPU,struct SDL::objectRanges& rangesInGPU)
{
    for(unsigned int lowmodIdx1=threadIdx.y+blockDim.y*blockIdx.y; lowmodIdx1<*(rangesInGPU.nEligibleT5Modules);lowmodIdx1+=gridDim.y*blockDim.y)
    {
        uint16_t lowmod1 = rangesInGPU.indicesOfEligibleT5Modules[lowmodIdx1];
        int nQuintuplets_lowmod1 = quintupletsInGPU.nQuintuplets[lowmod1];
        if(nQuintuplets_lowmod1==0) {continue;}
        int quintupletModuleIndices_lowmod1 = rangesInGPU.quintupletModuleIndices[lowmod1];

        for(unsigned int lowmodIdx2=threadIdx.x+blockDim.x*blockIdx.x; lowmodIdx2<*(rangesInGPU.nEligibleT5Modules);lowmodIdx2+=gridDim.x*blockDim.x)
        {
            uint16_t lowmod2 = rangesInGPU.indicesOfEligibleT5Modules[lowmodIdx2];
            int nQuintuplets_lowmod2 = quintupletsInGPU.nQuintuplets[lowmod2];
            if(nQuintuplets_lowmod2==0) {continue;}
            int quintupletModuleIndices_lowmod2 = rangesInGPU.quintupletModuleIndices[lowmod2];

            for(unsigned int ix1=0/*threadIdx.y*/; ix1<nQuintuplets_lowmod1; ix1+=1/*blockDim.y*/)
            {
                unsigned int ix = quintupletModuleIndices_lowmod1 + ix1;
                if(quintupletsInGPU.partOfPT5[ix] || quintupletsInGPU.isDup[ix]){continue;}

                for(unsigned int jx1=0; jx1<nQuintuplets_lowmod2; jx1++)
                {
                    unsigned int jx = quintupletModuleIndices_lowmod2 + jx1;
                    if(ix==jx){continue;}
                    if(quintupletsInGPU.partOfPT5[jx] || quintupletsInGPU.isDup[jx]){continue;}

                    float pt1  = __H2F(quintupletsInGPU.pt[ix]);
                    float eta1 = __H2F(quintupletsInGPU.eta[ix]);
                    float phi1 = __H2F(quintupletsInGPU.phi[ix]);
                    float score_rphisum1 = __H2F(quintupletsInGPU.score_rphisum[ix]);

                    float pt2  = __H2F(quintupletsInGPU.pt[jx]);
                    float eta2 = __H2F(quintupletsInGPU.eta[jx]);
                    float phi2 = __H2F(quintupletsInGPU.phi[jx]);
                    float score_rphisum2 = __H2F(quintupletsInGPU.score_rphisum[jx]);

                    float dEta = fabsf(eta1-eta2);
                    float dPhi = fabsf(phi1-phi2);

                    if (dEta > 0.1f){continue;}
                    if(dPhi > float(M_PI)){dPhi = dPhi - 2*float(M_PI);}
                    if (abs(dPhi) > 0.1f){continue;}
                    float dR2 = dEta*dEta + dPhi*dPhi;
                    int nMatched = checkHitsT5(ix,jx,quintupletsInGPU);
                    if(dR2 < 0.001f || nMatched >= 5)
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
                }
            }
        }
    }
}

__device__ float scorepT3(struct SDL::modules& modulesInGPU,struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU,struct SDL::segments& segmentsInGPU,struct SDL::triplets& tripletsInGPU, unsigned int innerPix, unsigned int outerTrip, float pt, float pz)
{
    unsigned int hits1[10];
    hits1[0] = mdsInGPU.anchorHitIndices[segmentsInGPU.mdIndices[2*innerPix]];
    hits1[1] = mdsInGPU.outerHitIndices[segmentsInGPU.mdIndices[2*innerPix]];
    hits1[2] = mdsInGPU.anchorHitIndices[segmentsInGPU.mdIndices[2*innerPix+1]];
    hits1[3] = mdsInGPU.outerHitIndices[segmentsInGPU.mdIndices[2*innerPix+1]];
    hits1[4] = mdsInGPU.anchorHitIndices[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip]]];
    hits1[5] = mdsInGPU.outerHitIndices[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip]]];
    hits1[6] = mdsInGPU.anchorHitIndices[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip]+1]];
    hits1[7] = mdsInGPU.outerHitIndices[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip]+1]];
    hits1[8] = mdsInGPU.anchorHitIndices[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip+1]+1]];
    hits1[9] = mdsInGPU.outerHitIndices[segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip+1]+1]];

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
        float var=0.f;
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
            err=0.15f*cosf(atanf(drdz));//(1.5mm)^2
        }
        else
        {
            err=5.0f*cosf(atanf(drdz));
        }//(5cm)^2
        score += (var*var) / (err*err);
    }
    //printf("pT3 score: %f\n",score);
    return score;
}
__device__ void inline checkHitspT3(unsigned int ix, unsigned int jx,struct SDL::pixelTriplets& pixelTripletsInGPU, int* matched)
{
    /*unsigned*/ int phits1[4] = {-1,-1,-1,-1};
    /*unsigned*/ int phits2[4] = {-1,-1,-1,-1};
    phits1[0] = pixelTripletsInGPU.hitIndices[10*ix];
    phits1[1] = pixelTripletsInGPU.hitIndices[10*ix+1];
    phits1[2] = pixelTripletsInGPU.hitIndices[10*ix+2];
    phits1[3] = pixelTripletsInGPU.hitIndices[10*ix+3];

    phits2[0] = pixelTripletsInGPU.hitIndices[10*jx];
    phits2[1] = pixelTripletsInGPU.hitIndices[10*jx+1];
    phits2[2] = pixelTripletsInGPU.hitIndices[10*jx+2];
    phits2[3] = pixelTripletsInGPU.hitIndices[10*jx+3];

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
    hits1[0] = pixelTripletsInGPU.hitIndices[10*ix+4];
    hits1[1] = pixelTripletsInGPU.hitIndices[10*ix+5];
    hits1[2] = pixelTripletsInGPU.hitIndices[10*ix+6];
    hits1[3] = pixelTripletsInGPU.hitIndices[10*ix+7];
    hits1[4] = pixelTripletsInGPU.hitIndices[10*ix+8];
    hits1[5] = pixelTripletsInGPU.hitIndices[10*ix+9];

    hits2[0] = pixelTripletsInGPU.hitIndices[10*jx+4];
    hits2[1] = pixelTripletsInGPU.hitIndices[10*jx+5];
    hits2[2] = pixelTripletsInGPU.hitIndices[10*jx+6];
    hits2[3] = pixelTripletsInGPU.hitIndices[10*jx+7];
    hits2[4] = pixelTripletsInGPU.hitIndices[10*jx+8];
    hits2[5] = pixelTripletsInGPU.hitIndices[10*jx+9];

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

    matched[0] = npMatched;
    matched[1] = nMatched;
}

__global__ void removeDupPixelTripletsInGPUFromMap(struct SDL::pixelTriplets& pixelTripletsInGPU, bool secondPass)
{
    for (unsigned int ix=blockIdx.y*blockDim.y+threadIdx.y; ix<*pixelTripletsInGPU.nPixelTriplets; ix+=blockDim.y*gridDim.y)
    {
        //bool isDup = false;
        float score1 = __H2F(pixelTripletsInGPU.score[ix]);
        for(unsigned int jx=blockIdx.x * blockDim.x + threadIdx.x; jx < *pixelTripletsInGPU.nPixelTriplets; jx += blockDim.x * gridDim.x)
        {
            float score2 = __H2F(pixelTripletsInGPU.score[jx]);
            if(ix==jx)
            {
                continue;
            }
            int nMatched[2];
            checkHitspT3(ix,jx,pixelTripletsInGPU,nMatched);
            if(((nMatched[0] + nMatched[1]) >= 5) )
            {
                //check the layers
                if(pixelTripletsInGPU.logicalLayers[5*jx+2] < pixelTripletsInGPU.logicalLayers[5*ix+2])
                {
                    rmPixelTripletToMemory(pixelTripletsInGPU, ix);
                    break;
                }
                else if( pixelTripletsInGPU.logicalLayers[5*ix+2] == pixelTripletsInGPU.logicalLayers[5*jx+2] && __H2F(pixelTripletsInGPU.score[ix]) > __H2F(pixelTripletsInGPU.score[jx]))
                {
                    rmPixelTripletToMemory(pixelTripletsInGPU,ix);
                    break;
                }
                else if( pixelTripletsInGPU.logicalLayers[5*ix+2] == pixelTripletsInGPU.logicalLayers[5*jx+2] && (__H2F(pixelTripletsInGPU.score[ix]) == __H2F(pixelTripletsInGPU.score[jx])) && (ix<jx))
                {
                    rmPixelTripletToMemory(pixelTripletsInGPU,ix);
                    break;
                }
            }
        }
    }
}



__global__ void removeDupPixelQuintupletsInGPUFromMap(struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, bool secondPass)
{
    int nPixelQuintuplets = *pixelQuintupletsInGPU.nPixelQuintuplets;

    for(unsigned int ix = blockIdx.y * blockDim.y + threadIdx.y; ix < nPixelQuintuplets; ix += blockDim.y * gridDim.y)
    {
        if(secondPass && pixelQuintupletsInGPU.isDup[ix])
        {
            continue;
        }
	    float score1 = __H2F(pixelQuintupletsInGPU.score[ix]);
        for(unsigned int jx = blockIdx.x * blockDim.x + threadIdx.x; jx < nPixelQuintuplets; jx += blockDim.x * gridDim.x)
        {
            if(ix == jx)
            {
                continue;
            }
            if(secondPass && pixelQuintupletsInGPU.isDup[jx])
            {
                continue;
            }
            int nMatched = checkHitspT5(ix, jx, pixelQuintupletsInGPU);
            float score2 = __H2F(pixelQuintupletsInGPU.score[jx]);
            if(nMatched >= 7)
            {
                if(score1 > score2 or ((score1 == score2) and (ix > jx)))
                {
                    rmPixelQuintupletToMemory(pixelQuintupletsInGPU, ix);
                    break;
                }
            }
        }
    }
}


__global__ void checkHitspLS(struct SDL::modules& modulesInGPU, struct SDL::objectRanges& rangesInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::hits& hitsInGPU,bool secondpass)
{
    int pixelModuleIndex = *modulesInGPU.nLowerModules;
    unsigned int nPixelSegments = segmentsInGPU.nSegments[pixelModuleIndex];
    if(nPixelSegments >  N_MAX_PIXEL_SEGMENTS_PER_MODULE)
    {
        nPixelSegments =  N_MAX_PIXEL_SEGMENTS_PER_MODULE;
    }
    for(int ix=blockIdx.y*blockDim.y+threadIdx.y;ix<nPixelSegments;ix+=blockDim.y*gridDim.y)
    {
        if(secondpass && (!segmentsInGPU.isQuad[ix] || segmentsInGPU.isDup[ix])){continue;}

        unsigned int phits1[4];  
        phits1[0] = segmentsInGPU.pLSHitsIdxs[ix].x;
        phits1[1] = segmentsInGPU.pLSHitsIdxs[ix].y;
        phits1[2] = segmentsInGPU.pLSHitsIdxs[ix].z;
        phits1[3] = segmentsInGPU.pLSHitsIdxs[ix].w;
        float eta_pix1 = segmentsInGPU.eta[ix];
        float phi_pix1 = segmentsInGPU.phi[ix];

        for(int jx = blockIdx.x * blockDim.x + threadIdx.x; jx < nPixelSegments; jx += blockDim.x * gridDim.x)
        {
            float eta_pix2 = segmentsInGPU.eta[jx];
            if (fabsf(eta_pix2 - eta_pix1) > 0.1f ) continue;

            if(secondpass && (!segmentsInGPU.isQuad[jx] || segmentsInGPU.isDup[jx])){continue;}
            if(ix==jx)
            {
                continue;
            }

            char quad_diff = segmentsInGPU.isQuad[ix] -segmentsInGPU.isQuad[jx];
            float ptErr_diff = segmentsInGPU.ptIn[ix] -segmentsInGPU.ptIn[jx];
            float score_diff = segmentsInGPU.score[ix] -segmentsInGPU.score[jx];
            if( (quad_diff > 0 )|| (score_diff<0 && quad_diff ==0))
            {
                continue;
            }// always keep quads over trips. If they are the same, we want the object with the lower pt Error

            unsigned int phits2[4];
            phits2[0] = segmentsInGPU.pLSHitsIdxs[jx].x;
            phits2[1] = segmentsInGPU.pLSHitsIdxs[jx].y;
            phits2[2] = segmentsInGPU.pLSHitsIdxs[jx].z;
            phits2[3] = segmentsInGPU.pLSHitsIdxs[jx].w;

            float phi_pix2 = segmentsInGPU.phi[jx];

            int npMatched =0;
            for (int i=0; i<4;i++)
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
                    if (secondpass) break; // only one hit is enough
                }
            }
            if((npMatched ==4) && (ix < jx))
            { // if exact match, remove only 1
                rmPixelSegmentFromMemory(segmentsInGPU,ix);
            }
            if(npMatched ==3)
            //if(npMatched >=2)
            {
                rmPixelSegmentFromMemory(segmentsInGPU,ix);
            }
            if(secondpass)
            {
                float dEta = abs(eta_pix1-eta_pix2);
                float dPhi = abs(phi_pix1-phi_pix2);
                if(dPhi > float(M_PI)){dPhi = dPhi - 2*float(M_PI);}
                float dR2 = dEta*dEta + dPhi*dPhi;

                if(npMatched >=1 or dR2 < 0.00075f and (ix < jx))
                {
                    rmPixelSegmentFromMemory(segmentsInGPU,ix); 
                }
            }
        }
    }
}

