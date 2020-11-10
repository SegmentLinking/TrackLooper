#ifndef HeterogeneousCore_CUDAUtilities_allocate_managed_h
#define HeterogeneousCore_CUDAUtilities_allocate_managed_h

#include <cuda_runtime.h>

namespace cms {
  namespace cuda {
    // Allocate managed memory (to be called from unique_ptr)
    //void *allocate_managed(unsigned int nbytes, cudaStream_t stream);
    void *allocate_managed(size_t nbytes, cudaStream_t stream);
    void *allocate_device(int dev, size_t nbytes, cudaStream_t stream);
    void *allocate_host(size_t nbytes, cudaStream_t stream);

    // Free managed memory (to be called from unique_ptr)
    void free_managed(void *ptr);
    void free_device(int dev, void *ptr);
    void free_host(void *ptr);
  }  // namespace cuda
}  // namespace cms

#endif
