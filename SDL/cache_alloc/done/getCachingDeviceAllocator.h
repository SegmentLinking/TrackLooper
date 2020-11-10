#ifndef HeterogeneousCore_CUDACore_src_getCachingDeviceAllocator
#define HeterogeneousCore_CUDACore_src_getCachingDeviceAllocator

#include <cuda_runtime.h>
#include "cuda_rt_call.h"
#include "CachingDeviceAllocator.h"

namespace cudautils {
  namespace allocator {
    // Use caching or not
    constexpr bool useCaching = true;
    // Growth factor (bin_growth in cub::CachingDeviceAllocator
    constexpr unsigned int binGrowth = 8;
    // Smallest bin, corresponds to binGrowth^minBin bytes (min_bin in cub::CacingDeviceAllocator
    constexpr unsigned int minBin = 1;
    // Largest bin, corresponds to binGrowth^maxBin bytes (max_bin in cub::CachingDeviceAllocator). Note that unlike in cub, allocations larger than binGrowth^maxBin are set to fail.
    constexpr unsigned int maxBin = 10;
    // Total storage for the allocator. 0 means no limit.
    constexpr size_t maxCachedBytes = 0;
    // Fraction of total device memory taken for the allocator. In case there are multiple devices with different amounts of memory, the smallest of them is taken. If maxCachedBytes is non-zero, the smallest of them is taken.
    constexpr double maxCachedFraction = 0.8;
    constexpr bool debug = false;

    inline size_t minCachedBytes() {
      size_t freeMemory, totalMemory;
      CUDA_RT_CALL(cudaMemGetInfo(&freeMemory, &totalMemory));
      size_t ret = static_cast<size_t>(maxCachedFraction * freeMemory);
      if (maxCachedBytes > 0) {
        ret = std::min(ret, maxCachedBytes);
      }
      return ret;
    }

    inline notcub::CachingDeviceAllocator& getCachingDeviceAllocator() {
      static notcub::CachingDeviceAllocator allocator{binGrowth,
                                                      minBin,
                                                      maxBin,
                                                      minCachedBytes(),
                                                      false,  // do not skip cleanup
                                                      debug};
      return allocator;
    }
  }  // namespace allocator
}  // namespace cudautils

#endif
