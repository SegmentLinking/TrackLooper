#ifndef Module_cuh
#define Module_cuh

#include <map>
#include <iostream>

#include "Constants.h"

namespace SDL {
  enum SubDet { InnerPixel = 0, Barrel = 5, Endcap = 4 };

  enum Side { NegZ = 1, PosZ = 2, Center = 3 };

  enum ModuleType { PS, TwoS, PixelModule };

  enum ModuleLayerType { Pixel, Strip, InnerPixelLayer };

  struct objectRanges {
    int* hitRanges;
    int* hitRangesLower;
    int* hitRangesUpper;
    int8_t* hitRangesnLower;
    int8_t* hitRangesnUpper;
    int* mdRanges;
    int* segmentRanges;
    int* trackletRanges;
    int* tripletRanges;
    int* trackCandidateRanges;
    // Others will be added later
    int* quintupletRanges;

    // This number is just nEligibleModules - 1, but still we want this to be independent of the TC kernel
    uint16_t* nEligibleT5Modules;
    // Will be allocated in createQuintuplets kernel!
    uint16_t* indicesOfEligibleT5Modules;
    // To store different starting points for variable occupancy stuff
    int* quintupletModuleIndices;
    int* quintupletModuleOccupancy;
    int* miniDoubletModuleIndices;
    int* miniDoubletModuleOccupancy;
    int* segmentModuleIndices;
    int* segmentModuleOccupancy;
    int* tripletModuleIndices;
    int* tripletModuleOccupancy;

    unsigned int* device_nTotalMDs;
    unsigned int* device_nTotalSegs;
    unsigned int* device_nTotalTrips;
    unsigned int* device_nTotalQuints;

    template <typename TBuff>
    void setData(TBuff& objectRangesbuf) {
      hitRanges = alpaka::getPtrNative(objectRangesbuf.hitRanges_buf);
      hitRangesLower = alpaka::getPtrNative(objectRangesbuf.hitRangesLower_buf);
      hitRangesUpper = alpaka::getPtrNative(objectRangesbuf.hitRangesUpper_buf);
      hitRangesnLower = alpaka::getPtrNative(objectRangesbuf.hitRangesnLower_buf);
      hitRangesnUpper = alpaka::getPtrNative(objectRangesbuf.hitRangesnUpper_buf);
      mdRanges = alpaka::getPtrNative(objectRangesbuf.mdRanges_buf);
      segmentRanges = alpaka::getPtrNative(objectRangesbuf.segmentRanges_buf);
      trackletRanges = alpaka::getPtrNative(objectRangesbuf.trackletRanges_buf);
      tripletRanges = alpaka::getPtrNative(objectRangesbuf.tripletRanges_buf);
      trackCandidateRanges = alpaka::getPtrNative(objectRangesbuf.trackCandidateRanges_buf);
      quintupletRanges = alpaka::getPtrNative(objectRangesbuf.quintupletRanges_buf);

      nEligibleT5Modules = alpaka::getPtrNative(objectRangesbuf.nEligibleT5Modules_buf);
      indicesOfEligibleT5Modules = alpaka::getPtrNative(objectRangesbuf.indicesOfEligibleT5Modules_buf);

      quintupletModuleIndices = alpaka::getPtrNative(objectRangesbuf.quintupletModuleIndices_buf);
      quintupletModuleOccupancy = alpaka::getPtrNative(objectRangesbuf.quintupletModuleOccupancy_buf);
      miniDoubletModuleIndices = alpaka::getPtrNative(objectRangesbuf.miniDoubletModuleIndices_buf);
      miniDoubletModuleOccupancy = alpaka::getPtrNative(objectRangesbuf.miniDoubletModuleOccupancy_buf);
      segmentModuleIndices = alpaka::getPtrNative(objectRangesbuf.segmentModuleIndices_buf);
      segmentModuleOccupancy = alpaka::getPtrNative(objectRangesbuf.segmentModuleOccupancy_buf);
      tripletModuleIndices = alpaka::getPtrNative(objectRangesbuf.tripletModuleIndices_buf);
      tripletModuleOccupancy = alpaka::getPtrNative(objectRangesbuf.tripletModuleOccupancy_buf);

      device_nTotalMDs = alpaka::getPtrNative(objectRangesbuf.device_nTotalMDs_buf);
      device_nTotalSegs = alpaka::getPtrNative(objectRangesbuf.device_nTotalSegs_buf);
      device_nTotalTrips = alpaka::getPtrNative(objectRangesbuf.device_nTotalTrips_buf);
      device_nTotalQuints = alpaka::getPtrNative(objectRangesbuf.device_nTotalQuints_buf);
    }
  };

  template <typename TAcc>
  struct objectRangesBuffer : objectRanges {
    Buf<TAcc, int> hitRanges_buf;
    Buf<TAcc, int> hitRangesLower_buf;
    Buf<TAcc, int> hitRangesUpper_buf;
    Buf<TAcc, int8_t> hitRangesnLower_buf;
    Buf<TAcc, int8_t> hitRangesnUpper_buf;
    Buf<TAcc, int> mdRanges_buf;
    Buf<TAcc, int> segmentRanges_buf;
    Buf<TAcc, int> trackletRanges_buf;
    Buf<TAcc, int> tripletRanges_buf;
    Buf<TAcc, int> trackCandidateRanges_buf;
    Buf<TAcc, int> quintupletRanges_buf;

    Buf<TAcc, uint16_t> nEligibleT5Modules_buf;
    Buf<TAcc, uint16_t> indicesOfEligibleT5Modules_buf;

    Buf<TAcc, int> quintupletModuleIndices_buf;
    Buf<TAcc, int> quintupletModuleOccupancy_buf;
    Buf<TAcc, int> miniDoubletModuleIndices_buf;
    Buf<TAcc, int> miniDoubletModuleOccupancy_buf;
    Buf<TAcc, int> segmentModuleIndices_buf;
    Buf<TAcc, int> segmentModuleOccupancy_buf;
    Buf<TAcc, int> tripletModuleIndices_buf;
    Buf<TAcc, int> tripletModuleOccupancy_buf;

    Buf<TAcc, unsigned int> device_nTotalMDs_buf;
    Buf<TAcc, unsigned int> device_nTotalSegs_buf;
    Buf<TAcc, unsigned int> device_nTotalTrips_buf;
    Buf<TAcc, unsigned int> device_nTotalQuints_buf;

    template <typename TQueue, typename TDevAcc>
    objectRangesBuffer(unsigned int nMod, unsigned int nLowerMod, TDevAcc const& devAccIn, TQueue& queue)
        : hitRanges_buf(allocBufWrapper<int>(devAccIn, nMod * 2, queue)),
          hitRangesLower_buf(allocBufWrapper<int>(devAccIn, nMod, queue)),
          hitRangesUpper_buf(allocBufWrapper<int>(devAccIn, nMod, queue)),
          hitRangesnLower_buf(allocBufWrapper<int8_t>(devAccIn, nMod, queue)),
          hitRangesnUpper_buf(allocBufWrapper<int8_t>(devAccIn, nMod, queue)),
          mdRanges_buf(allocBufWrapper<int>(devAccIn, nMod * 2, queue)),
          segmentRanges_buf(allocBufWrapper<int>(devAccIn, nMod * 2, queue)),
          trackletRanges_buf(allocBufWrapper<int>(devAccIn, nMod * 2, queue)),
          tripletRanges_buf(allocBufWrapper<int>(devAccIn, nMod * 2, queue)),
          trackCandidateRanges_buf(allocBufWrapper<int>(devAccIn, nMod * 2, queue)),
          quintupletRanges_buf(allocBufWrapper<int>(devAccIn, nMod * 2, queue)),
          nEligibleT5Modules_buf(allocBufWrapper<uint16_t>(devAccIn, 1, queue)),
          indicesOfEligibleT5Modules_buf(allocBufWrapper<uint16_t>(devAccIn, nLowerMod, queue)),
          quintupletModuleIndices_buf(allocBufWrapper<int>(devAccIn, nLowerMod, queue)),
          quintupletModuleOccupancy_buf(allocBufWrapper<int>(devAccIn, nLowerMod, queue)),
          miniDoubletModuleIndices_buf(allocBufWrapper<int>(devAccIn, nLowerMod + 1, queue)),
          miniDoubletModuleOccupancy_buf(allocBufWrapper<int>(devAccIn, nLowerMod + 1, queue)),
          segmentModuleIndices_buf(allocBufWrapper<int>(devAccIn, nLowerMod + 1, queue)),
          segmentModuleOccupancy_buf(allocBufWrapper<int>(devAccIn, nLowerMod + 1, queue)),
          tripletModuleIndices_buf(allocBufWrapper<int>(devAccIn, nLowerMod, queue)),
          tripletModuleOccupancy_buf(allocBufWrapper<int>(devAccIn, nLowerMod, queue)),
          device_nTotalMDs_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          device_nTotalSegs_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          device_nTotalTrips_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          device_nTotalQuints_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)) {
      alpaka::memset(queue, hitRanges_buf, -1, nMod * 2);
      alpaka::memset(queue, hitRangesLower_buf, -1, nMod);
      alpaka::memset(queue, hitRangesUpper_buf, -1, nMod);
      alpaka::memset(queue, hitRangesnLower_buf, -1, nMod);
      alpaka::memset(queue, hitRangesnUpper_buf, -1, nMod);
      alpaka::memset(queue, mdRanges_buf, -1, nMod * 2);
      alpaka::memset(queue, segmentRanges_buf, -1, nMod * 2);
      alpaka::memset(queue, trackletRanges_buf, -1, nMod * 2);
      alpaka::memset(queue, tripletRanges_buf, -1, nMod * 2);
      alpaka::memset(queue, trackCandidateRanges_buf, -1, nMod * 2);
      alpaka::memset(queue, quintupletRanges_buf, -1, nMod * 2);
      alpaka::memset(queue, quintupletModuleIndices_buf, -1, nLowerMod);
      alpaka::wait(queue);
    }
  };

  struct modules {
    unsigned int* detIds;
    uint16_t* moduleMap;
    unsigned int* mapdetId;
    uint16_t* mapIdx;
    uint16_t* nConnectedModules;
    float* drdzs;
    float* slopes;
    uint16_t* nModules;
    uint16_t* nLowerModules;
    uint16_t* partnerModuleIndices;

    short* layers;
    short* rings;
    short* modules;
    short* rods;
    short* subdets;
    short* sides;
    float* eta;
    float* r;
    bool* isInverted;
    bool* isLower;
    bool* isAnchor;
    ModuleType* moduleType;
    ModuleLayerType* moduleLayerType;
    int* sdlLayers;

    unsigned int* connectedPixels;

    bool parseIsInverted(short subdet, short side, short module, short layer) {
      if (subdet == Endcap) {
        if (side == NegZ) {
          return module % 2 == 1;
        } else if (side == PosZ) {
          return module % 2 == 0;
        } else {
          return 0;
        }
      } else if (subdet == Barrel) {
        if (side == Center) {
          if (layer <= 3) {
            return module % 2 == 1;
          } else if (layer >= 4) {
            return module % 2 == 0;
          } else {
            return 0;
          }
        } else if (side == NegZ or side == PosZ) {
          if (layer <= 2) {
            return module % 2 == 1;
          } else if (layer == 3) {
            return module % 2 == 0;
          } else {
            return 0;
          }
        } else {
          return 0;
        }
      } else {
        return 0;
      }
    };

    bool parseIsLower(bool isInvertedx, unsigned int detId) { return (isInvertedx) ? !(detId & 1) : (detId & 1); };

    unsigned int parsePartnerModuleId(unsigned int detId, bool isLowerx, bool isInvertedx) {
      return isLowerx ? (isInvertedx ? detId - 1 : detId + 1) : (isInvertedx ? detId + 1 : detId - 1);
    };

    template <typename TBuff>
    void setData(TBuff& modulesbuf) {
      detIds = alpaka::getPtrNative(modulesbuf.detIds_buf);
      moduleMap = alpaka::getPtrNative(modulesbuf.moduleMap_buf);
      mapdetId = alpaka::getPtrNative(modulesbuf.mapdetId_buf);
      mapIdx = alpaka::getPtrNative(modulesbuf.mapIdx_buf);
      nConnectedModules = alpaka::getPtrNative(modulesbuf.nConnectedModules_buf);
      drdzs = alpaka::getPtrNative(modulesbuf.drdzs_buf);
      slopes = alpaka::getPtrNative(modulesbuf.slopes_buf);
      nModules = alpaka::getPtrNative(modulesbuf.nModules_buf);
      nLowerModules = alpaka::getPtrNative(modulesbuf.nLowerModules_buf);
      partnerModuleIndices = alpaka::getPtrNative(modulesbuf.partnerModuleIndices_buf);

      layers = alpaka::getPtrNative(modulesbuf.layers_buf);
      rings = alpaka::getPtrNative(modulesbuf.rings_buf);
      modules = alpaka::getPtrNative(modulesbuf.modules_buf);
      rods = alpaka::getPtrNative(modulesbuf.rods_buf);
      subdets = alpaka::getPtrNative(modulesbuf.subdets_buf);
      sides = alpaka::getPtrNative(modulesbuf.sides_buf);
      eta = alpaka::getPtrNative(modulesbuf.eta_buf);
      r = alpaka::getPtrNative(modulesbuf.r_buf);
      isInverted = alpaka::getPtrNative(modulesbuf.isInverted_buf);
      isLower = alpaka::getPtrNative(modulesbuf.isLower_buf);
      isAnchor = alpaka::getPtrNative(modulesbuf.isAnchor_buf);
      moduleType = alpaka::getPtrNative(modulesbuf.moduleType_buf);
      moduleLayerType = alpaka::getPtrNative(modulesbuf.moduleLayerType_buf);

      connectedPixels = alpaka::getPtrNative(modulesbuf.connectedPixels_buf);
      sdlLayers = alpaka::getPtrNative(modulesbuf.sdlLayers_buf);
    }
  };

  template <typename TAcc>
  struct modulesBuffer : modules {
    Buf<TAcc, unsigned int> detIds_buf;
    Buf<TAcc, uint16_t> moduleMap_buf;
    Buf<TAcc, unsigned int> mapdetId_buf;
    Buf<TAcc, uint16_t> mapIdx_buf;
    Buf<TAcc, uint16_t> nConnectedModules_buf;
    Buf<TAcc, float> drdzs_buf;
    Buf<TAcc, float> slopes_buf;
    Buf<TAcc, uint16_t> nModules_buf;
    Buf<TAcc, uint16_t> nLowerModules_buf;
    Buf<TAcc, uint16_t> partnerModuleIndices_buf;

    Buf<TAcc, short> layers_buf;
    Buf<TAcc, short> rings_buf;
    Buf<TAcc, short> modules_buf;
    Buf<TAcc, short> rods_buf;
    Buf<TAcc, short> subdets_buf;
    Buf<TAcc, short> sides_buf;
    Buf<TAcc, float> eta_buf;
    Buf<TAcc, float> r_buf;
    Buf<TAcc, bool> isInverted_buf;
    Buf<TAcc, bool> isLower_buf;
    Buf<TAcc, bool> isAnchor_buf;
    Buf<TAcc, ModuleType> moduleType_buf;
    Buf<TAcc, ModuleLayerType> moduleLayerType_buf;

    Buf<TAcc, unsigned int> connectedPixels_buf;
    Buf<TAcc, int> sdlLayers_buf;

    template <typename TDevAcc>
    modulesBuffer(TDevAcc const& devAccIn, unsigned int nMod = modules_size, unsigned int nPixs = pix_tot)
      : detIds_buf(allocBufWrapper<unsigned int>(devAccIn, nMod)),
        moduleMap_buf(allocBufWrapper<uint16_t>(devAccIn, nMod * 40)),
        mapdetId_buf(allocBufWrapper<unsigned int>(devAccIn, nMod)),
        mapIdx_buf(allocBufWrapper<uint16_t>(devAccIn, nMod)),
        nConnectedModules_buf(allocBufWrapper<uint16_t>(devAccIn, nMod)),
        drdzs_buf(allocBufWrapper<float>(devAccIn, nMod)),
        slopes_buf(allocBufWrapper<float>(devAccIn, nMod)),
        nModules_buf(allocBufWrapper<uint16_t>(devAccIn, 1)),
        nLowerModules_buf(allocBufWrapper<uint16_t>(devAccIn, 1)),
        partnerModuleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, nMod)),
        
        layers_buf(allocBufWrapper<short>(devAccIn, nMod)),
        rings_buf(allocBufWrapper<short>(devAccIn, nMod)),
        modules_buf(allocBufWrapper<short>(devAccIn, nMod)),
        rods_buf(allocBufWrapper<short>(devAccIn, nMod)),
        subdets_buf(allocBufWrapper<short>(devAccIn, nMod)),
        sides_buf(allocBufWrapper<short>(devAccIn, nMod)),
        eta_buf(allocBufWrapper<float>(devAccIn, nMod)),
        r_buf(allocBufWrapper<float>(devAccIn, nMod)),
        isInverted_buf(allocBufWrapper<bool>(devAccIn, nMod)),
        isLower_buf(allocBufWrapper<bool>(devAccIn, nMod)),
        isAnchor_buf(allocBufWrapper<bool>(devAccIn, nMod)),
        moduleType_buf(allocBufWrapper<ModuleType>(devAccIn, nMod)),
        moduleLayerType_buf(allocBufWrapper<ModuleLayerType>(devAccIn, nMod)),
        sdlLayers_buf(allocBufWrapper<int>(devAccIn, nMod)),
        
        connectedPixels_buf(allocBufWrapper<unsigned int>(devAccIn, nPixs)) {}
    
    template<typename TQueue>
    inline void copyFromSrc(TQueue queue, const modulesBuffer<alpaka::DevCpu>& src) {
      alpaka::memcpy(queue, detIds_buf, src.detIds_buf);
      alpaka::memcpy(queue, moduleMap_buf, src.moduleMap_buf);
      alpaka::memcpy(queue, mapdetId_buf, src.mapdetId_buf);
      alpaka::memcpy(queue, mapIdx_buf, src.mapIdx_buf);
      alpaka::memcpy(queue, nConnectedModules_buf, src.nConnectedModules_buf);
      alpaka::memcpy(queue, drdzs_buf, src.drdzs_buf);
      alpaka::memcpy(queue, slopes_buf, src.slopes_buf);
      alpaka::memcpy(queue, nModules_buf, src.nModules_buf);
      alpaka::memcpy(queue, nLowerModules_buf, src.nLowerModules_buf);
      alpaka::memcpy(queue, partnerModuleIndices_buf, src.partnerModuleIndices_buf);
      
      alpaka::memcpy(queue, layers_buf, src.layers_buf);
      alpaka::memcpy(queue, rings_buf, src.rings_buf);
      alpaka::memcpy(queue, modules_buf, src.modules_buf);
      alpaka::memcpy(queue, rods_buf, src.rods_buf);
      alpaka::memcpy(queue, subdets_buf, src.subdets_buf);
      alpaka::memcpy(queue, sides_buf, src.sides_buf);
      alpaka::memcpy(queue, eta_buf, src.eta_buf);
      alpaka::memcpy(queue, r_buf, src.r_buf);
      alpaka::memcpy(queue, isInverted_buf, src.isInverted_buf);
      alpaka::memcpy(queue, isLower_buf, src.isLower_buf);
      alpaka::memcpy(queue, isAnchor_buf, src.isAnchor_buf);
      alpaka::memcpy(queue, moduleType_buf, src.moduleType_buf);
      alpaka::memcpy(queue, moduleLayerType_buf, src.moduleLayerType_buf);
      alpaka::memcpy(queue, sdlLayers_buf, src.sdlLayers_buf);
      
      alpaka::memcpy(queue, connectedPixels_buf, src.connectedPixels_buf);
      alpaka::wait(queue);
    }

    template<typename TQueue>
    modulesBuffer(TQueue queue, const modulesBuffer<alpaka::DevCpu>& src,
                  unsigned int nMod = modules_size,
                  unsigned int nPixs = pix_tot)
      : modulesBuffer(alpaka::getDev(queue), nMod, nPixs)
    {
      copyFromSrc(queue, src);
    }
    
  };


}  // namespace SDL
#endif
