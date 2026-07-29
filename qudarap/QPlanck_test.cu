/**
QPlanck_test.cu
=================

See qudarap/QPlanck_test.h for the interface to use the
below extern "C" interface from non-nvcc compiled objects.

**/


#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "qplanck.h"

__global__ void setup_curand_kernel(curandState_t* states, unsigned long seed, size_t num_values) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_values) {
        curand_init(seed, id, 0, &states[id]);
    }
}

template <typename T>
__global__ void _qplanck_test_kernel(curandState_t* states, T* d_values, size_t num_values, qplanck* planck)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_values) return ;

    float u = curand_uniform(&states[id]);
    d_values[id] = planck->wavelength(u);
}

template <typename T>
static void _qplanck_test(T* h_values, size_t num_values, unsigned long seed, qplanck* planck) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_values + threadsPerBlock - 1) / threadsPerBlock;

    curandState_t* d_states = nullptr;
    T* d_values = nullptr;

    cudaMalloc(&d_states, num_values * sizeof(curandState_t));
    cudaMalloc(&d_values, num_values * sizeof(T));

    setup_curand_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, seed, num_values);

    _qplanck_test_kernel<T><<<blocksPerGrid, threadsPerBlock>>>(d_states, d_values, num_values, planck );

    cudaDeviceSynchronize();

    cudaMemcpy(h_values, d_values, num_values * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    cudaFree(d_states);
}

// Concrete C-interface symbols for GCC
extern "C" {

void qplanck_test_float(float* values, size_t num_values, unsigned long seed, qplanck* d_planck) {
    _qplanck_test<float>(values, num_values, seed, d_planck);
}

void qplanck_test_double(double* values, size_t num_values, unsigned long seed, qplanck* d_planck) {
    _qplanck_test<double>(values, num_values, seed, d_planck);
}

} // extern "C"
