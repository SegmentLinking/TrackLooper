#include <limits>

#include "cuda_rt_call.h"
#include "allocate_device.h"
#include "getCachingDeviceAllocator.h"
#include <iostream>

namespace {
  const size_t maxAllocationSize =
      notcub::CachingDeviceAllocator::IntPow(cudautils::allocator::binGrowth, cudautils::allocator::maxBin);
}

namespace cudautils {
  void *allocate_device(int dev, size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    if (nbytes > maxAllocationSize) {
      std::cout<<"at stream"<<stream<<std::endl;
      throw std::runtime_error("alloate_device : Tried to allocate " + std::to_string(nbytes) +
                               " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
    }
    CUDA_RT_CALL(cudautils::allocator::getCachingDeviceAllocator().DeviceAllocate(dev, &ptr, nbytes, stream));
    return ptr;
  }

  void free_device(int device, void *ptr) {
    CUDA_RT_CALL(cudautils::allocator::getCachingDeviceAllocator().DeviceFree(device, ptr));
  }

}  // namespace cudautils
