#include <cuda_runtime.h>
#include "smath.h"


template <typename T>
__global__ void log_kernel(T* d_values, T* d_domain, size_t num_values, T x0, T x1 ) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_values) return ;

    T x = x0 + (x1-x0)*T(id)/T(num_values-1);
    d_domain[id] = x ;
    d_values[id] = smath::log<T>(x);
}

template <typename T>
static void _launch_log_kernel(T* h_values, T* h_domain, size_t num_values, T x0, T x1) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_values + threadsPerBlock - 1) / threadsPerBlock;

    T* d_values = nullptr;
    T* d_domain = nullptr;
    cudaMalloc(&d_values, num_values * sizeof(T));
    cudaMalloc(&d_domain, num_values * sizeof(T));

    log_kernel<T><<<blocksPerGrid, threadsPerBlock>>>(d_values, d_domain, num_values, x0, x1);
    cudaDeviceSynchronize();

    cudaMemcpy(h_values, d_values, num_values * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_domain, d_domain, num_values * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    cudaFree(d_domain);
}

// Using a C-interface for symbols crossing between objects compiled by gcc and nvcc avoids potential issues
extern "C" {
void launch_log_kernel_float(  float* values, float* domain,  size_t num_values, float x0, float x1) {     _launch_log_kernel<float>(values, domain,  num_values, x0, x1); }
void launch_log_kernel_double(double* values, double* domain, size_t num_values, double x0, double x1) {   _launch_log_kernel<double>(values, domain, num_values, x0, x1); }
} // extern "C"
