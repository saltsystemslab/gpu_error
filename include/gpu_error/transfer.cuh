#ifndef GPU_ERROR_TRANSFER_CUH
#define GPU_ERROR_TRANSFER_CUH

#include <cuda_runtime.h>

namespace gpu_error {

// Constructor wrapper to deduce type from assignment
template<typename SizeType>
struct host_constructor {
    SizeType size_param;
    
    __host__ host_constructor(SizeType size) : size_param(size) {}
    
    template<typename T>
    __host__ operator T*() {
        T* host_ptr;
        cudaMallocHost(&host_ptr, sizeof(T) * size_param);
        return host_ptr;
    }
};

// Get host version of an object (allocate on host)
template<typename SizeType = int>
__host__ host_constructor<SizeType> get_host_version(SizeType size_param = 1) {
    return host_constructor<SizeType>(size_param);
}

// Constructor wrapper for device allocation with size parameter
template<typename SizeType>
struct device_constructor {
    SizeType size_param;
    
    __host__ device_constructor(SizeType size) : size_param(size) {}
    
    template<typename T>
    __host__ operator T*() {
        T* device_ptr;
        cudaMalloc(&device_ptr, sizeof(T) * size_param);
        return device_ptr;
    }
};

// Get device version of an object (allocate on device)
template<typename SizeType = int>
__host__ device_constructor<SizeType> get_device_version(SizeType size_param = 1) {
    return device_constructor<SizeType>(size_param);
}

// Copy object to device
template<typename T>
__host__ T* copy_to_device(T* host_ptr) {
    T* device_ptr;
    cudaMalloc(&device_ptr, sizeof(T));
    cudaMemcpy(device_ptr, host_ptr, sizeof(T), cudaMemcpyHostToDevice);
    return device_ptr;
}

// Copy object to host
template<typename T>
__host__ T* copy_to_host(T* device_ptr) {
    T* host_ptr;
    cudaMallocHost(&host_ptr, sizeof(T));
    cudaMemcpy(host_ptr, device_ptr, sizeof(T), cudaMemcpyDeviceToHost);
    return host_ptr;
}

// Move object to device (copy and free host)
template<typename T>
__host__ T* move_to_device(T* host_ptr) {
    T* device_ptr = copy_to_device(host_ptr);
    cudaFreeHost(host_ptr);
    return device_ptr;
}

// Move object to host (copy and free device)
template<typename T>
__host__ T* move_to_host(T* device_ptr) {
    T* host_ptr = copy_to_host(device_ptr);
    cudaFree(device_ptr);
    return host_ptr;
}

} // namespace gpu_error

#endif // GPU_ERROR_TRANSFER_CUH
