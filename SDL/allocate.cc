#include <limits>

#include "allocate.h"
#include "cudaCheck.h"

#include "getCachingAllocator.h"

namespace {
  const size_t maxAllocationSize =
      notcub::CachingDeviceAllocator::IntPow(cms::cuda::allocator::binGrowth, cms::cuda::allocator::maxBin);
}

namespace cms::cuda {
 // void *allocate_managed(unsigned int nbytes, cudaStream_t stream) {
  void *allocate_managed(size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
//    if constexpr (allocator::useCaching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      cudaCheck(allocator::getCachingManagedAllocator().ManagedAllocate(&ptr, nbytes, stream));
//    } else {
//      cudaCheck(cudaMallocManaged(&ptr, nbytes));
//    }
    return ptr;
  }

  void free_managed(void *ptr) {
    //if constexpr (allocator::useCaching) {
      cudaCheck(allocator::getCachingManagedAllocator().ManagedFree(ptr));
    //} else {
    //  cudaCheck(cudaFree(ptr));
    //}
  }

  void *allocate_device(int dev, size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    if (nbytes > maxAllocationSize) {
      std::cout<<"at stream"<<stream<<std::endl;
      throw std::runtime_error("alloate_device : Tried to allocate " + std::to_string(nbytes) +
                               " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
    }
    cudaCheck(allocator::getCachingDeviceAllocator().DeviceAllocate(dev, &ptr, nbytes, stream));
    return ptr;
  }

  void free_device(int device, void *ptr) {
    cudaCheck(allocator::getCachingDeviceAllocator().DeviceFree(device, ptr));
  }

  void *allocate_host(size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    if (nbytes > maxAllocationSize) {
      throw std::runtime_error("allocate_host: Tried to allocate " + std::to_string(nbytes) +
                               " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
    }
    cudaCheck(allocator::getCachingHostAllocator().HostAllocate(&ptr, nbytes, stream));
    return ptr;
  }

  void free_host(void *ptr) {
    cudaCheck(allocator::getCachingHostAllocator().HostFree(ptr));
  }

}  // namespace cms::cuda
