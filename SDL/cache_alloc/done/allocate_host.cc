#include <limits>

#include "cuda_rt_call.h"
#include "getCachingHostAllocator.h"

namespace {
  const size_t maxAllocationSize =
      notcub::CachingDeviceAllocator::IntPow(cudautils::allocator::binGrowth, cudautils::allocator::maxBin);
}

namespace cudautils {
  void *allocate_host(size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    if (nbytes > maxAllocationSize) {
      throw std::runtime_error("allocate_host: Tried to allocate " + std::to_string(nbytes) +
                               " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
    }
    CUDA_RT_CALL(cudautils::allocator::getCachingHostAllocator().HostAllocate(&ptr, nbytes, stream));
    return ptr;
  }

  void free_host(void *ptr) {
    CUDA_RT_CALL(cudautils::allocator::getCachingHostAllocator().HostFree(ptr));
  }

}  // namespace cudautils
