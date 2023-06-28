#ifndef HeterogeneousCore_AlpakaInterface_interface_getHostCachingAllocator_h
#define HeterogeneousCore_AlpakaInterface_interface_getHostCachingAllocator_h

#include "thread_safety_macros.h"
#include "AllocatorConfig.h"
#include "CachingAllocator.h"
#include "config.h"
#include "host.h"
#include "traits.h"

namespace cms::alpakatools {

  template <typename TQueue, typename = std::enable_if_t<cms::alpakatools::is_queue_v<TQueue>>>
  inline CachingAllocator<alpaka_common::DevHost, TQueue>& getHostCachingAllocator() {
    // thread safe initialisation of the host allocator
    CMS_THREAD_SAFE static CachingAllocator<alpaka_common::DevHost, TQueue> allocator(
        host(),
        config::binGrowth,
        config::minBin,
        config::maxBin,
        config::maxCachedBytes,
        config::maxCachedFraction,
        false,   // reuseSameQueueAllocations
        false);  // debug

    // the public interface is thread safe
    return allocator;
  }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_getHostCachingAllocator_h
